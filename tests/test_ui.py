"""Presentation regression checks; no model downloads required."""

import unittest

import gradio as gr

from aksara_ui import EXAMPLES, build_ui, render_context


class PresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls = []
        cls.result = (
            "soalan asal",
            "1. (0.9876)  Makan puji — Suka dipuji.",
            "Suka dipuji.",
            "2.34s",
            "Model: test | LoRA: ON",
        )

        def backend(*args):
            cls.calls.append(args)
            if args[0] == "fail":
                raise RuntimeError("backend detail")
            return cls.result

        cls.demo = build_ui(
            backend,
            ["Makan puji"],
            lambda n: f"Apakah maksud peribahasa {n}?" if n else "",
            "test",
            lambda: False,
        )
        cls.submit = next(
            fn for fn in cls.demo.fns.values() if fn.api_name == "tanya_aksara"
        )

    def test_original_five_outputs_and_control_values_are_preserved(self):
        states = list(self.submit.fn("Soalan", 5, False, True))
        self.assertEqual(self.calls[-1], ("Soalan", 5, False, True))
        self.assertEqual(states[-1][:5], self.result)
        self.assertEqual(len(states[-1]), len(self.submit.outputs))
        self.assertIn("Sumber 1", states[-1][5])
        self.assertIn("0.9876", states[-1][5])
        self.assertIn("2.34s", states[-1][6])
        self.assertIn("Tidak aktif", states[-1][6])
        self.assertEqual(states[0][:5], ("", "", "", "", ""))

    def test_all_example_callbacks_fill_their_own_question(self):
        callbacks = list(self.demo.fns.values())[:4]
        self.assertEqual([fn.fn() for fn in callbacks], EXAMPLES)

    def test_context_is_escaped_and_multiline_content_is_preserved(self):
        raw = "1. (0.9981)  Tajuk — <script>alert(1)</script>\nBaris kedua\n2. (-0.0123)  Teks tanpa tajuk"
        rendered = render_context(raw)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("Baris kedua", rendered)
        self.assertIn("Teks tanpa tajuk", rendered)
        self.assertEqual(rendered.count('class="context-card"'), 2)

    def test_blank_question_does_not_call_backend(self):
        count = len(self.calls)
        with self.assertRaises(gr.Error):
            list(self.submit.fn("  ", 1, True, True))
        self.assertEqual(len(self.calls), count)

    def test_failure_clears_outputs_and_preserves_debug_logging(self):
        request = self.submit.fn("fail", 1, True, False)
        next(request)
        with self.assertLogs("aksara_ui", level="ERROR") as logs:
            failed = next(request)
        self.assertEqual(failed[:5], ("", "", "", "", ""))
        self.assertIn("tidak dapat dijana", failed[6])
        self.assertIn("backend detail", " ".join(logs.output))
        with self.assertRaises(gr.Error):
            next(request)

    def test_submit_and_enter_share_model_concurrency_limit(self):
        events = list(self.demo.fns.values())[-2:]
        self.assertEqual([fn.concurrency_id for fn in events], ["aksara-model"] * 2)
        self.assertEqual([fn.concurrency_limit for fn in events], [1, 1])
        self.assertEqual(events[0].outputs, events[1].outputs)


if __name__ == "__main__":
    unittest.main()
