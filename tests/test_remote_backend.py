import os
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from aksara_ui import render_status
from remote_backend import answer_remote


class RemoteBackendTests(unittest.TestCase):
    def test_forwards_all_controls_and_preserves_five_outputs(self):
        result = ("question", "context", "answer", "0.25s", "LoRA: ON")
        client = Mock()
        client.predict.return_value = result
        with patch("remote_backend.get_client", return_value=client):
            self.assertEqual(answer_remote("question", 5, False, True), result)
        client.predict.assert_called_once_with(
            "question", 5, False, True, api_name="/answer"
        )

    def test_rejects_changed_remote_contract(self):
        client = Mock()
        client.predict.return_value = ("incomplete",)
        with (
            patch("remote_backend.get_client", return_value=client),
            self.assertRaises(RuntimeError),
        ):
            answer_remote("question", 1, True, False)

    def test_api_status_does_not_claim_adapter_is_loaded(self):
        self.assertIn(
            "Tidak disahkan oleh API",
            render_status("Jawapan tersedia", "0.25s", 1, None),
        )

    def test_api_startup_does_not_import_local_inference_stack(self):
        env = dict(os.environ, AKSARA_BACKEND="space")
        env.pop("SPACE_ID", None)
        check = subprocess.run(
            [
                sys.executable,
                "-c",
                "import app, sys; assert app.BACKEND == 'space'; assert not ({'torch', 'faiss', 'transformers', 'sentence_transformers', 'peft'} & set(sys.modules))",
            ],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(check.returncode, 0, check.stderr)


if __name__ == "__main__":
    unittest.main()
