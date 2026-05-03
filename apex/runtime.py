"""Toy MoE runtime, with optional async prefetch + learned predictor.

Walks through 2 layers, asks the router which experts to use at each
layer, makes sure each chosen expert is loaded, runs it via the backend,
and averages the per-expert outputs to form the next hidden state.

Optional pluggables:
  - prefetcher: schedules next-layer expert loads in the background
    after each layer's compute, then `wait_for_layer` is called before
    the next layer to make those loads visible as cache hits.
  - predictor: guesses next-layer experts from the current layer's
    expert set; if it has no prediction (returns []) the runtime falls
    back to peeking the router for the actual next-layer experts.
  - trace_recorder: writes the (current -> actual-next) transition to
    JSONL so a future predictor can be trained on real usage.

Backend-native hidden state (step 2):
  The hidden state is converted to the backend's preferred form once at
  the start (`backend.prepare_hidden`) and stays in that form across
  layers (`backend.combine_outputs` averages per-layer outputs in
  place). Only at two boundaries do we cross back to numpy:
    - calling the router (numpy routers want ndarray; MLX router doesn't
      care because it accepts either via internal handling)
    - returning the final hidden from `run()` (callers / tests expect
      ndarray)
  This means under MLXBackend, every intermediate matmul result is an
  mx.array — no per-layer round trip through numpy.
"""

import numpy as np


class Runtime:
    def __init__(
        self,
        router,
        expert_manager,
        backend,
        prefetcher=None,
        predictor=None,
        trace_recorder=None,
        num_layers: int = 2,
    ):
        # Decides which experts run at each layer (ground truth).
        # Routers must accept (layer_id, hidden_state) — FakeRouter
        # ignores the hidden_state argument, GateRouter uses it.
        self.router = router
        # Owns the cache + store; gives us loaded expert weights.
        self.expert_manager = expert_manager
        # Compute backend. Provides prepare_hidden / run_expert /
        # combine_outputs / to_numpy so the runtime stays
        # array-type-agnostic (numpy vs mx.array).
        self.backend = backend
        # Optional: AsyncPrefetcher for background next-layer loading.
        self.prefetcher = prefetcher
        # Optional: MarkovPredictor (or anything with .predict(...)).
        self.predictor = predictor
        # Optional: TraceRecorder that captures real transitions to disk.
        self.trace_recorder = trace_recorder
        # Number of layers in the toy model. Default 2 keeps every
        # existing demo unchanged; future real-model layouts can pass
        # whatever the model actually has.
        self.num_layers = num_layers

    def run(self):
        # Start with a dummy hidden state (batch=1, hidden=8) and let
        # the backend transform it into its preferred form. For
        # FakeBackend this is a no-op; for MLXBackend we now hold an
        # mx.array that travels through every layer without round trips.
        hidden = self.backend.prepare_hidden(
            np.ones((1, 8), dtype="float32")
        )

        for layer_id in range(self.num_layers):
            # If we prefetched experts for this layer at the end of the
            # previous one, wait for those background loads to finish so
            # the ensure_loaded calls below become guaranteed cache hits.
            if self.prefetcher is not None and layer_id > 0:
                self.prefetcher.wait_for_layer(layer_id)

            print(f"running layer {layer_id}")

            # Ask the router which experts to use for this layer.
            # Routers handle either ndarray or mx.array internally
            # (numpy routers call np.asarray, MLX router wraps with
            # mx.array). So we pass `hidden` directly and let the
            # router decide what to do with it.
            expert_ids = self.router.route(layer_id, hidden)
            # `router.name` lets demos see which routing path is in
            # use ("numpy" vs "mlx"). All routers declare a name.
            print(f"router backend: {self.router.name}")
            print(f"router selected experts: {list(expert_ids)}")

            # Run each chosen expert and collect its output.
            outputs = []
            for expert_id in expert_ids:
                # ensure_loaded handles cache hit / miss / eviction and
                # (if backend was supplied) returns a backend-prepared
                # tensor — for MLXBackend, that's an mx.array.
                expert = self.expert_manager.ensure_loaded(layer_id, expert_id)
                # The backend applies the expert's weights to the hidden
                # state. With MLXBackend this is mx.matmul; both inputs
                # are already mx.array (hidden from prepare_hidden,
                # expert from prepare_expert in the manager).
                out = self.backend.run_expert(hidden, expert)
                outputs.append(out)

            # Combine the per-expert outputs by simple averaging. The
            # backend owns the math so mx.array stays mx.array (no
            # implicit numpy conversion via Python's sum() pitfalls).
            hidden = self.backend.combine_outputs(outputs)

            # End-of-layer side work: trace recording + prefetch scheduling.
            # Doing prefetch AFTER current-layer compute (rather than in
            # parallel) avoids the prefetch evicting an expert that
            # current-layer compute is still using — important when the
            # memory budget is tight.
            next_layer = layer_id + 1
            if next_layer < self.num_layers:
                # Peek the router so we know what *actually* runs next.
                # The recorder needs this; the prefetch fallback uses it
                # if the predictor has no opinion. Pass the freshly
                # computed `hidden` so a gate router scores the same
                # state the next layer will see.
                actual_next_experts = self.router.route(next_layer, hidden)

                if self.trace_recorder is not None:
                    self.trace_recorder.record(
                        layer_id, expert_ids, next_layer, actual_next_experts
                    )

                if self.predictor is not None:
                    predicted = self.predictor.predict(
                        current_layer=layer_id,
                        current_experts=expert_ids,
                        next_layer=next_layer,
                        k=2,
                    )
                else:
                    predicted = []

                if predicted:
                    print(f"predictor predicted next layer experts: {list(predicted)}")
                    prefetch_targets = predicted
                elif self.predictor is not None:
                    # Predictor was attached but had no prediction —
                    # surface the fallback so users can see the predictor
                    # missing on this transition.
                    print(
                        f"predictor returned no prediction; falling back to router: "
                        f"{list(actual_next_experts)}"
                    )
                    prefetch_targets = actual_next_experts
                else:
                    # No predictor wired up at all — same fallback path.
                    prefetch_targets = actual_next_experts

                if self.prefetcher is not None:
                    self.prefetcher.prefetch_many(next_layer, prefetch_targets)

        # Return the final hidden as numpy regardless of backend so
        # callers / tests have a stable contract.
        return self.backend.to_numpy(hidden)
