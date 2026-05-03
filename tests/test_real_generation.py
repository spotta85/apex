"""Real-Gemma end-to-end test for step 16E.

One real layer of Gemma 4 26B-A4B-IT NVFP4: embed token 2 → layer 0
decoder → final norm → tied lm_head + softcap → argmax. Asserts the
predicted next-token id is in vocab range and logits are finite.

Skips when the checkpoint isn't on disk; embedding dequant alone
takes a few seconds and ~3 GB of RAM, so the always-on synthetic
suite stays in tests/test_generation.py.
"""

import contextlib

import numpy as np

from apex.backends.fake_backend import FakeBackend
from apex.decoder import single_token_decoder_layer
from apex.dequant import DequantizingExpertStore
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.generation import embed_token, final_logits, sample_argmax
from apex.metrics import Metrics
from apex.moe_layer import run_moe_layer
from apex.ops import rms_norm
from apex.real_router import MultiLayerRealRouter
from tests._synthetic_nvfp4 import skip_if_no_real_checkpoint


@contextlib.contextmanager
def _silence_matmul_warnings():
    with np.errstate(over="ignore", under="ignore",
                     invalid="ignore", divide="ignore"):
        yield


@skip_if_no_real_checkpoint()
def test_real_gemma_one_token_forward_emits_in_vocab_id():
    """Embed → 1 decoder layer → final_norm → lm_head → argmax."""
    from apex.checkpoint_loaders import (
        load_attention_weights,
        load_embedding,
        load_final_norm,
        load_layer_norms,
        read_text_config,
    )
    from scripts._real_moe_helpers import quantized_store_for_layout
    from tests._synthetic_nvfp4 import (
        real_checkpoint_path,
        real_layout_and_quant_store,
    )

    path = real_checkpoint_path()
    layout, _ = real_layout_and_quant_store(path)
    text_cfg = read_text_config(path)

    embedding = load_embedding(path, layout)
    final_norm_w = load_final_norm(path, layout)
    assert embedding.shape == (text_cfg["vocab_size"], text_cfg["hidden_size"])
    assert final_norm_w.shape == (text_cfg["hidden_size"],)

    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=512, verbose=False, metrics=metrics)
    quant_store = quantized_store_for_layout(path, layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()
    router = MultiLayerRealRouter(path, layout, top_k=8)

    attn_weights = load_attention_weights(path, layout, layer_id=0)
    norms = load_layer_norms(path, layout, layer_id=0)
    attn_kwargs = dict(
        q_weight=attn_weights["q_weight"], k_weight=attn_weights["k_weight"],
        v_weight=attn_weights["v_weight"], o_weight=attn_weights["o_weight"],
        num_heads=text_cfg["num_attention_heads"],
        num_kv_heads=text_cfg["num_key_value_heads"],
        head_dim=text_cfg["head_dim"],
        q_norm_weight=attn_weights["q_norm"],
        k_norm_weight=attn_weights["k_norm"],
        rms_eps=text_cfg["rms_norm_eps"],
    )

    def moe_call(h):
        with _silence_matmul_warnings():
            ids, weights = router.route(0, h)
            return run_moe_layer(
                h, ids, weights,
                lambda eid: manager.ensure_loaded(0, eid),
                backend,
            )

    # Token id 2 = Gemma 4 BOS per text_config.bos_token_id.
    token_id = 2
    hidden = embed_token(embedding, token_id)
    assert hidden.shape == (1, text_cfg["hidden_size"])

    with _silence_matmul_warnings():
        hidden = single_token_decoder_layer(
            hidden,
            norms=norms,
            attn_kwargs=attn_kwargs,
            moe_call=moe_call,
            rms_eps=text_cfg["rms_norm_eps"],
        )
    assert np.all(np.isfinite(hidden))

    hidden = rms_norm(
        hidden, final_norm_w,
        eps=text_cfg["rms_norm_eps"], gemma_unit_offset=True,
    )
    assert np.all(np.isfinite(hidden))

    softcap = text_cfg["final_logit_softcapping"]
    with _silence_matmul_warnings():
        logits = final_logits(hidden, embedding, softcap=softcap)

    assert logits.shape == (1, text_cfg["vocab_size"])
    assert np.all(np.isfinite(logits))
    # Softcap must clamp logits to within ±softcap. Mathematically the
    # bound is open (tanh ∈ (-1, 1)) but fp32 saturates at exactly
    # 1.0, so the floating-point bound is closed. Without the cap a
    # raw logit on real Gemma can easily exceed 30 in magnitude.
    assert np.all(np.abs(logits) <= softcap), float(np.abs(logits).max())

    next_id = sample_argmax(logits)
    assert 0 <= next_id < text_cfg["vocab_size"], next_id
