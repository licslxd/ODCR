"""step5_engine：import 不触发 T5 加载；懒加载与测试注入覆盖。"""
import importlib
import os
import sys
import unittest

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from paths_config import get_step5_text_model_dir  # noqa: E402


class _DummyTok:
    """最小 tokenizer 桩：仅用于验证覆盖路径不触碰 HF。"""

    name_or_path = "dummy_tok"
    model_max_length = 512
    eos_token_id = 1

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3]}

    def batch_decode(self, ids, skip_special_tokens=True):
        del ids, skip_special_tokens
        return ["ok"]


class TestStep5EngineTokenizerLazy(unittest.TestCase):
    def tearDown(self) -> None:
        mod = sys.modules.get("executors.step5_engine")
        if mod is None:
            return
        if hasattr(mod, "set_step5_tokenizer_override"):
            mod.set_step5_tokenizer_override(None)
        if hasattr(mod, "_step5_tokenizer_obj"):
            mod._step5_tokenizer_obj = None  # type: ignore[attr-defined]

    def test_import_without_loading_real_tokenizer(self) -> None:
        self.tearDown()
        if "executors.step5_engine" in sys.modules:
            del sys.modules["executors.step5_engine"]
        m = importlib.import_module("executors.step5_engine")
        self.assertIsNone(getattr(m, "_step5_tokenizer_obj", object()))

    def test_override_skips_hf_load(self) -> None:
        if "executors.step5_engine" in sys.modules:
            del sys.modules["executors.step5_engine"]
        m = importlib.import_module("executors.step5_engine")
        m.set_step5_tokenizer_override(_DummyTok())
        tok = m.get_step5_tokenizer()
        self.assertIsInstance(tok, _DummyTok)
        self.assertIsNone(m._step5_tokenizer_obj)

    @unittest.skipUnless(
        os.path.isdir(get_step5_text_model_dir()),
        "无本地 google__flan-t5-xl 目录时跳过（离线 fail-fast 环境须自备 tokenizer）",
    )
    def test_lazy_path_returns_same_instance_after_first_load(self) -> None:
        """第二次 get_step5_tokenizer() 复用同一缓存对象。"""
        if "executors.step5_engine" in sys.modules:
            del sys.modules["executors.step5_engine"]
        m = importlib.import_module("executors.step5_engine")
        m.set_step5_tokenizer_override(None)
        m._step5_tokenizer_obj = None
        a = m.get_step5_tokenizer()
        b = m.get_step5_tokenizer()
        self.assertIs(a, b)


class TestIndexContractDualOnlyReadable(unittest.TestCase):
    def test_schema_version_constant(self) -> None:
        from odcr_core import index_contract as ic

        self.assertEqual(ic.INDEX_CONTRACT_SCHEMA_VERSION, "odcr_index_contract/2.2")


if __name__ == "__main__":
    unittest.main()
