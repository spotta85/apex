"""Public APEX API.

Two-line quickstart:

    from apex import MoEModel
    model = MoEModel.from_pretrained("/path/to/checkpoint", max_memory_gb=8)
    text = model.generate("Explain MoE simply.", max_tokens=64)
    print(text)

Everything else (cache, manager, router, backends) is internal — most
callers should never need to touch it. The `apex.*` submodules remain
importable for advanced users that want to compose lower-level pieces.
"""

from apex.config import ModelConfig
from apex.kv_cache import KVCache
from apex.memory import KVMemoryReport, MemoryReport, RuntimeMemoryReport
from apex.model import MoEModel
from apex.tokenizer import Tokenizer

__all__ = [
    "MoEModel",
    "ModelConfig",
    "Tokenizer",
    "KVCache",
    "MemoryReport",
    "KVMemoryReport",
    "RuntimeMemoryReport",
]
