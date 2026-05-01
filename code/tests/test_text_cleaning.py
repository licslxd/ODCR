"""odcr_core.text_cleaning 单测（unittest，无 pytest 依赖）。"""
import os
import sys
import unittest

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from odcr_core.text_cleaning import (  # noqa: E402
    build_sample_quality_flags,
    build_template_stats,
    clean_explanation_text,
    detect_bad_tail,
    detect_short_fragment,
    html_entity_hit_raw,
)


class TestTextCleaning(unittest.TestCase):
    def test_html_unescape(self) -> None:
        r = clean_explanation_text("I love &amp; hate this &lt;movie&gt;")
        self.assertIn("love", r.clean_text)
        self.assertIn("&", r.clean_text)

    def test_strip_incomplete_entity_tail(self) -> None:
        r = clean_explanation_text("Good film but ends with &frac1")
        self.assertFalse(r.clean_text.endswith("&"))

    def test_bad_tail_unclosed_paren(self) -> None:
        d = detect_bad_tail("This is (unfinished")
        self.assertTrue(d["bad_tail_hit"])
        self.assertIn("unclosed_paren", d["bad_tail_types"])

    def test_short_fragment(self) -> None:
        self.assertTrue(detect_short_fragment("a b"))
        self.assertFalse(detect_short_fragment("one two three four"))

    def test_template_hit(self) -> None:
        stats = build_template_stats(["hello world", "hello world", "hello world", "other"])
        flags = build_sample_quality_flags(
            raw_explanation="hello world",
            clean_result=clean_explanation_text("hello world"),
            template_stats=stats,
            template_min_count=3,
        )
        self.assertTrue(flags["template_hit"])

    def test_html_entity_hit_raw(self) -> None:
        self.assertTrue(html_entity_hit_raw("x &amp; y"))
        self.assertFalse(html_entity_hit_raw("no entity here"))


if __name__ == "__main__":
    unittest.main()
