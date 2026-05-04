"""Step 25C: long-context behavior — sliding-window scope + KV linearity.

Step 24C added the `window_size` kwarg to `attention_decode_step` and
proved (synthetically) that the slice happens at small cache lengths.
What it did NOT prove is what 25C cares about:

  1. **At long context** (cache_len ≫ window), entries OLDER than
     the window must have NO effect on attention output. Without
     this, the window is a memory-layout no-op rather than a
     correctness fix for >1024-token prompts.

  2. **Without a window** (full_attention layers, `window_size=None`),
     those same old entries DO affect output — proves the absence of
     a default slice.

  3. **KV cache memory grows linearly** with sequence length. APEX's
     `KVCache` is intentionally unbounded (no rolling window — see
     `apex/kv_cache.py` docstring). Locking that in here means a
     future "drop the cache when it exceeds X" change has to update
     this test, surfacing the policy decision instead of silently
     growing memory cost.

The "long" scale used here is synthetic (window=64, sequence~300) —
the math of the slice is invariant in the cache size, so a synthetic
proof is sufficient. A real-Gemma >1024-token prefill end-to-end is
deferred (the current python decode loop is O(prompt_len) so a real
1500-token prompt at L=8 takes minutes per test invocation).
"""

import contextlib

import numpy as np

from apex.attention import attention_decode_step
from apex.kv_cache import KVCache


@contextlib.contextmanager
def _silence_matmul_warnings():
    """Same pattern as test_real_attention — random matmuls overflow even
    when the final output is finite."""
    with np.errstate(over="ignore", under="ignore",
                     invalid="ignore", divide="ignore"):
        yield


def _attn_kwargs(rng, hidden_dim, num_heads, num_kv_heads, head_dim):
    return dict(
        q_weight=rng.standard_normal(
            (num_heads * head_dim, hidden_dim),
        ).astype(np.float32),
        k_weight=rng.standard_normal(
            (num_kv_heads * head_dim, hidden_dim),
        ).astype(np.float32),
        v_weight=rng.standard_normal(
            (num_kv_heads * head_dim, hidden_dim),
        ).astype(np.float32),
        o_weight=rng.standard_normal(
            (hidden_dim, num_heads * head_dim),
        ).astype(np.float32),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def test_old_entries_outside_window_do_not_affect_attention_output():
    """Under windowed attention, corruption of pre-window entries is invisible.

    Build a long cache (N=200 entries), run windowed attention
    (window=64) on the new query, snapshot the output. Then corrupt
    the oldest 100 entries (well outside the window) and re-run; the
    output must be byte-identical because those entries are out of
    scope. If the slice is a no-op, the corruption would change the
    softmax distribution and the output would drift.
    """
    rng = np.random.default_rng(100)
    hidden_dim = 32
    num_heads, num_kv_heads, head_dim = 4, 2, 8
    window = 64
    n_entries = 200  # ~3x the window — well into "long context" regime

    kwargs = _attn_kwargs(rng, hidden_dim, num_heads, num_kv_heads, head_dim)
    hidden = rng.standard_normal((1, hidden_dim)).astype(np.float32)

    # Build a cache with n_entries by direct injection. Bypassing the
    # decode loop keeps the test fast and lets us mutate specific
    # ranges without re-running attention. K/V shape is what
    # KVCache.append would have produced after n_entries calls.
    cache_a = KVCache(num_layers=1)
    cache_b = KVCache(num_layers=1)

    full_k = rng.standard_normal(
        (1, num_kv_heads, n_entries, head_dim),
    ).astype(np.float32)
    full_v = rng.standard_normal(
        (1, num_kv_heads, n_entries, head_dim),
    ).astype(np.float32)
    cache_a.keys[0] = full_k.copy()
    cache_a.values[0] = full_v.copy()
    cache_b.keys[0] = full_k.copy()
    cache_b.values[0] = full_v.copy()

    # Corrupt cache_b's first 100 entries — well outside the last-64
    # window. If windowing works, attention won't see this.
    cache_b.keys[0][..., :100, :] = 999.0
    cache_b.values[0][..., :100, :] = -999.0

    with _silence_matmul_warnings():
        out_a = attention_decode_step(
            hidden, **kwargs, kv_cache=cache_a, layer_id=0,
            window_size=window,
        )
        out_b = attention_decode_step(
            hidden, **kwargs, kv_cache=cache_b, layer_id=0,
            window_size=window,
        )

    np.testing.assert_array_equal(out_a, out_b)


def test_old_entries_do_affect_output_when_window_is_none():
    """Same setup, no window ⇒ corruption changes the output.

    Counterpart to the test above: with `window_size=None`, the
    slice doesn't happen and ALL cache entries contribute to the
    softmax. Same corruption must now produce a different output.
    Together the two tests prove the slice is doing exactly what
    its semantics claim.
    """
    rng = np.random.default_rng(101)
    hidden_dim = 32
    num_heads, num_kv_heads, head_dim = 4, 2, 8
    n_entries = 200

    kwargs = _attn_kwargs(rng, hidden_dim, num_heads, num_kv_heads, head_dim)
    hidden = rng.standard_normal((1, hidden_dim)).astype(np.float32)

    cache_a = KVCache(num_layers=1)
    cache_b = KVCache(num_layers=1)

    full_k = rng.standard_normal(
        (1, num_kv_heads, n_entries, head_dim),
    ).astype(np.float32)
    full_v = rng.standard_normal(
        (1, num_kv_heads, n_entries, head_dim),
    ).astype(np.float32)
    cache_a.keys[0] = full_k.copy()
    cache_a.values[0] = full_v.copy()
    cache_b.keys[0] = full_k.copy()
    cache_b.values[0] = full_v.copy()
    # Corrupt cache_b's first 100 entries.
    cache_b.keys[0][..., :100, :] = 999.0
    cache_b.values[0][..., :100, :] = -999.0

    with _silence_matmul_warnings():
        out_a = attention_decode_step(
            hidden, **kwargs, kv_cache=cache_a, layer_id=0,
            # window_size omitted ⇒ None ⇒ full attention scope
        )
        out_b = attention_decode_step(
            hidden, **kwargs, kv_cache=cache_b, layer_id=0,
        )

    # Outputs must be meaningfully different — the corruption is large
    # (999) so softmax weights shift visibly. `not np.allclose` with
    # default tolerance is the right check (we expect non-trivial drift,
    # not just last-bit noise).
    assert not np.allclose(out_a, out_b), (
        "full-attention scope must include all cache entries; "
        "corruption of 100 old entries left the output unchanged "
        "— the slice is happening even with window_size=None"
    )


def test_attention_finite_at_long_sequence_lengths():
    """At cache_len=300 with windowed attention, output stays finite.

    Long sequences exercise the softmax over a large axis: the
    pre-softmax score range is wider, denormals more likely,
    overflow more probable than at the small-seq tests in 24C. This
    test pins that the windowed read keeps things stable at
    O(window) entries regardless of cache total size — by design,
    only `window_size` rows feed softmax.
    """
    rng = np.random.default_rng(102)
    hidden_dim = 32
    num_heads, num_kv_heads, head_dim = 4, 2, 8
    window = 128
    n_entries = 300

    kwargs = _attn_kwargs(rng, hidden_dim, num_heads, num_kv_heads, head_dim)
    hidden = rng.standard_normal((1, hidden_dim)).astype(np.float32)

    cache = KVCache(num_layers=1)
    cache.keys[0] = rng.standard_normal(
        (1, num_kv_heads, n_entries, head_dim),
    ).astype(np.float32)
    cache.values[0] = rng.standard_normal(
        (1, num_kv_heads, n_entries, head_dim),
    ).astype(np.float32)

    with _silence_matmul_warnings():
        out = attention_decode_step(
            hidden, **kwargs, kv_cache=cache, layer_id=0,
            window_size=window,
        )

    assert out.shape == (1, hidden_dim)
    assert np.all(np.isfinite(out))
    # Window did NOT trim the cache — only the read view. Cache still
    # has all original entries plus the appended query.
    assert cache.length(0) == n_entries + 1


def test_kv_cache_total_bytes_grows_linearly_with_appends():
    """KVCache memory ∝ seq length × per-token-bytes.

    Locks in the current "no rolling window in the cache" policy:
    appending N tokens to layer L grows total_bytes by exactly
    N × (2 × num_kv_heads × head_dim × bytes_per_elem). Sliding-
    window attention is a READ-time slice, not a cache trim — this
    test would fail if a future change starts evicting old entries
    from the cache itself, surfacing the policy change instead of
    silently dropping context.
    """
    cache = KVCache(num_layers=1)
    num_kv_heads, head_dim = 2, 8

    # Bytes per (k, v) pair at one position: each side is
    # (1, num_kv_heads, 1, head_dim) float32 = 1*2*1*8*4 = 64 bytes.
    # Both sides ⇒ 128 bytes per appended token.
    bytes_per_token = 2 * num_kv_heads * head_dim * 4

    for step in (1, 5, 50, 200, 1024, 1500):
        # Reset for a clean linearity measurement at each step count.
        cache.reset()
        for _ in range(step):
            k = np.zeros(
                (1, num_kv_heads, 1, head_dim), dtype=np.float32,
            )
            v = np.zeros(
                (1, num_kv_heads, 1, head_dim), dtype=np.float32,
            )
            cache.append(0, k, v)
        assert cache.total_bytes == step * bytes_per_token, (
            f"at seq={step}: total_bytes={cache.total_bytes}, "
            f"expected={step * bytes_per_token}"
        )
        assert cache.length(0) == step


def test_kv_cache_holds_well_past_sliding_window_size():
    """KVCache happily stores 1500+ entries without hitting any internal cap.

    Defends against a future change that adds a "rolling window of
    sliding_window entries" to KVCache — which would silently drop
    older context for full_attention layers (they need the full
    history). The test forces 1500 appends (>Gemma 4's
    sliding_window=1024) and asserts the cache still holds all of
    them. If we ever DO add KV trimming, this test must be updated
    to express the new contract — not deleted.
    """
    cache = KVCache(num_layers=1)
    num_kv_heads, head_dim = 2, 8
    n = 1500
    for _ in range(n):
        k = np.zeros((1, num_kv_heads, 1, head_dim), dtype=np.float32)
        v = np.zeros((1, num_kv_heads, 1, head_dim), dtype=np.float32)
        cache.append(0, k, v)

    assert cache.length(0) == n
    keys, values = cache.get(0)
    assert keys.shape == (1, num_kv_heads, n, head_dim)
    assert values.shape == (1, num_kv_heads, n, head_dim)
