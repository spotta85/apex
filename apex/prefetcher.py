"""Background prefetcher: loads experts for a future layer while the
runtime is still working on the current one.

The architecture is intentionally minimal for the MVP:

  Runtime finishes layer L
       |
       v
  prefetcher.prefetch_many(L+1, expert_ids)  --> queue
                                                  |
                                                  v
                                          background worker
                                                  |
                                                  v
                                       expert_manager.ensure_loaded
                                       (source="prefetch" so it doesn't
                                       print "cache miss" lines)
                                                  |
                                                  v
                                           cache populated
       |
       v
  Runtime asks for layer L+1 experts -> cache hits

`wait_for_layer(L)` lets the runtime block until all in-flight loads
*for that layer* are done, which makes the demo trace deterministic
(every layer L+1 lookup is a guaranteed hit). Real-world usage would
typically *not* wait — the whole point of prefetching is to overlap
disk I/O with compute. We keep the wait here so the MVP is easy to
reason about; dropping it later just turns this into best-effort
overlap with a graceful fallback to a normal blocking load.

`verbose=False` suppresses the per-load "prefetch worker: loading"
line (used by the predictor demo for a focused trace). The
"scheduled prefetch" line always prints — it's the visible signal of
the prediction step itself.

Dedupe via `_in_flight` so requesting the same expert twice doesn't
queue two loads. Already-cached experts are skipped at schedule time.
"""

import queue
import threading


class AsyncPrefetcher:
    def __init__(self, expert_manager, verbose: bool = True, metrics=None):
        # The manager owns the cache + store. We delegate the actual
        # load through it so cache accounting / eviction stays in one
        # place — we just call it with source="prefetch" so the worker's
        # output is its own "prefetch worker: loading" line.
        self.expert_manager = expert_manager

        # Work queue. queue.Queue is already thread-safe, so we don't
        # need an extra lock around it.
        self._queue: "queue.Queue" = queue.Queue()

        # Set of (layer, expert) currently scheduled or being loaded.
        # Guarded by _state_lock along with _done_events.
        self._in_flight: set = set()

        # Per-key Event so wait_for_layer can block until the worker
        # actually finishes the load (not just dequeues it).
        self._done_events: dict = {}

        self._state_lock = threading.Lock()

        # When False, the worker doesn't print its own "loading" line.
        self.verbose = verbose

        # Optional Metrics; bumped on schedule + on completion.
        self.metrics = metrics

        # daemon=True so a forgotten shutdown() doesn't hang the
        # program at interpreter exit.
        self._shutdown = False
        self._worker = threading.Thread(target=self._run, daemon=True, name="apex-prefetch")
        self._worker.start()

    def schedule(self, layer_id: int, expert_id: int):
        """Queue one expert for background loading. Idempotent + dedup'd."""
        key = (layer_id, expert_id)

        with self._state_lock:
            # Skip if already loaded. Cheap check to avoid pointless work.
            if self.expert_manager.cache.has(key):
                return
            # Skip if a worker is already going to load this.
            if key in self._in_flight:
                return
            self._in_flight.add(key)
            # Create the completion event before we publish the work,
            # so wait_for_layer() can never see "in flight but no event".
            self._done_events[key] = threading.Event()

        self._queue.put(key)
        # Always print: this is the visible "we made a prediction" signal.
        print(f"scheduled prefetch: layer {layer_id} expert {expert_id}")
        if self.metrics is not None:
            self.metrics.on_prefetch_scheduled(key)

    def prefetch_many(self, layer_id: int, expert_ids):
        """Convenience: schedule a whole layer's worth of experts."""
        for expert_id in expert_ids:
            self.schedule(layer_id, expert_id)

    def wait_for_layer(self, layer_id: int):
        """Block until every prefetch we've scheduled for `layer_id` finishes.

        Used by the runtime right before computing layer L so the
        ensure_loaded calls in that layer's compute step are guaranteed
        cache hits (assuming we prefetched them). Skipping this call is
        legal — it just means lookups race with the worker and may fall
        back to a normal blocking load.
        """
        with self._state_lock:
            # Snapshot the events so we can wait without holding the
            # lock (workers need the lock to mark themselves done).
            events = [event for key, event in self._done_events.items() if key[0] == layer_id]

        for event in events:
            event.wait()

    def shutdown(self):
        """Stop the worker thread cleanly. Safe to call multiple times."""
        self._shutdown = True
        # Sentinel pushes the worker out of its blocking get().
        self._queue.put(None)
        self._worker.join(timeout=2)

    def _run(self):
        # Worker loop. Pulls (layer, expert) tuples from the queue and
        # loads each one. None is the shutdown sentinel.
        while True:
            key = self._queue.get()
            if key is None or self._shutdown:
                break

            layer_id, expert_id = key
            if self.verbose:
                print(f"prefetch worker: loading layer {layer_id} expert {expert_id}")

            # Do the load via the manager so cache accounting + eviction
            # behave identically to a foreground miss. source="prefetch"
            # so the manager doesn't double up on miss/hit print lines
            # (we already printed "prefetch worker: loading" above) and
            # doesn't bump foreground cache_hit/miss counters.
            self.expert_manager.ensure_loaded(layer_id, expert_id, source="prefetch")

            # Signal completion AFTER the load is in the cache, so a
            # waiter that wakes up will see the cache populated.
            with self._state_lock:
                self._in_flight.discard(key)
                event = self._done_events.get(key)
                if event is not None:
                    event.set()

            if self.metrics is not None:
                self.metrics.on_prefetch_completed(key)
