"""Tests for apex/kv_cache.py — append/get/length/total_bytes/reset."""

import numpy as np
import pytest

from apex.kv_cache import KVCache


def _kv(seq_len: int, fill: float = 0.0):
    """Build a (1, 2, seq_len, 4) ndarray (batch=1, kv_heads=2, head_dim=4)."""
    return np.full((1, 2, seq_len, 4), fill, dtype=np.float32)


def test_empty_cache_has_zero_length_and_zero_bytes():
    cache = KVCache(num_layers=3)
    for layer_id in range(3):
        assert cache.length(layer_id) == 0
        keys, values = cache.get(layer_id)
        assert keys is None and values is None
    assert cache.total_bytes == 0


def test_first_append_initializes_layer_buffers():
    cache = KVCache(num_layers=2)
    k, v = _kv(1, 1.0), _kv(1, 2.0)
    cache.append(0, k, v)
    keys, values = cache.get(0)
    np.testing.assert_array_equal(keys, k)
    np.testing.assert_array_equal(values, v)
    assert cache.length(0) == 1
    # Other layers untouched.
    assert cache.length(1) == 0
    assert cache.get(1) == (None, None)


def test_subsequent_appends_concatenate_along_seq_axis():
    cache = KVCache(num_layers=1)
    cache.append(0, _kv(1, 1.0), _kv(1, 1.0))
    cache.append(0, _kv(1, 2.0), _kv(1, 2.0))
    cache.append(0, _kv(2, 3.0), _kv(2, 3.0))

    assert cache.length(0) == 4
    keys, values = cache.get(0)
    assert keys.shape == (1, 2, 4, 4)
    assert values.shape == (1, 2, 4, 4)
    # First two seq positions are 1.0, third is 2.0, last two are 3.0.
    np.testing.assert_array_equal(keys[0, 0, 0], 1.0)
    np.testing.assert_array_equal(keys[0, 0, 1], 2.0)
    np.testing.assert_array_equal(keys[0, 0, 2], 3.0)
    np.testing.assert_array_equal(keys[0, 0, 3], 3.0)


def test_per_layer_isolation():
    """Appending to one layer must not affect any other."""
    cache = KVCache(num_layers=3)
    cache.append(1, _kv(2), _kv(2))
    assert cache.length(0) == 0
    assert cache.length(1) == 2
    assert cache.length(2) == 0


def test_total_bytes_sums_keys_and_values_across_layers():
    cache = KVCache(num_layers=2)
    cache.append(0, _kv(1), _kv(1))
    # (1, 2, 1, 4) = 8 fp32 elements = 32 bytes per tensor; 2 tensors = 64.
    assert cache.total_bytes == 64
    cache.append(1, _kv(3), _kv(3))
    # Layer 1 added (1, 2, 3, 4) = 24 elements × 4 bytes × 2 tensors = 192.
    assert cache.total_bytes == 64 + 192


def test_reset_drops_all_cached_state():
    cache = KVCache(num_layers=2)
    cache.append(0, _kv(2), _kv(2))
    cache.append(1, _kv(3), _kv(3))
    assert cache.total_bytes > 0
    cache.reset()
    assert cache.total_bytes == 0
    for layer_id in range(2):
        assert cache.length(layer_id) == 0
        assert cache.get(layer_id) == (None, None)


def test_invalid_layer_id_raises_index_error():
    cache = KVCache(num_layers=2)
    with pytest.raises(IndexError):
        cache.append(2, _kv(1), _kv(1))
    with pytest.raises(IndexError):
        cache.length(-1)
    with pytest.raises(IndexError):
        cache.get(5)


def test_zero_or_negative_num_layers_raises():
    with pytest.raises(ValueError):
        KVCache(num_layers=0)
    with pytest.raises(ValueError):
        KVCache(num_layers=-1)


def test_append_with_mismatched_kv_shape_raises():
    """k and v must share batch/heads/head_dim — only seq dim varies."""
    cache = KVCache(num_layers=1)
    k = _kv(1)
    v_wrong = np.zeros((1, 2, 1, 8), dtype=np.float32)  # head_dim 8 vs k's 4
    with pytest.raises(ValueError):
        cache.append(0, k, v_wrong)
