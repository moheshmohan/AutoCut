"""Failure-path regressions; fake tools write owned fixture bytes, never run RAEA/GPU."""
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

loader = importlib.util.spec_from_file_location("raea_handoff", Path(__file__).with_name("render.py"))
handoff = importlib.util.module_from_spec(loader)
loader.loader.exec_module(handoff)


class RenderRecovery(unittest.TestCase):
    def fixture(self, root):
        source = root / "scene.tsx"
        source.write_text("// generated fixture; renderer is mocked\n")
        return SimpleNamespace(source=source, output=root / "revision", preview=False, spec=None,
                               size=None, canvas=None, dependency=[], raea="/mock/raea", cli="/mock/cli",
                               composition="scene", alpha=False, fps=None, cache=False)

    @staticmethod
    def render_bytes(args, **kwargs):
        Path(args[args.index("-o") + 1]).write_bytes(b"completed render")

    def assert_retained(self, failed_probe):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            failure = subprocess.CalledProcessError(1, "native QA")
            probe = patch.object(handoff.subprocess, "check_output", side_effect=failure) if failed_probe else patch.object(
                handoff.subprocess, "check_output", return_value=json.dumps({"video": {"width": 960, "height": 540}}))
            with patch.object(handoff.subprocess, "run", side_effect=self.render_bytes), probe, patch.object(handoff, "command", side_effect=failure):
                with self.assertRaises(subprocess.CalledProcessError):
                    handoff.render(args)
            self.assertEqual((args.output / "revision.mp4").read_bytes(), b"completed render")
            self.assertFalse((args.output / "autocut-import.json").exists())

    def test_completed_render_survives_probe_failure(self):
        self.assert_retained(True)

    def test_completed_render_survives_preflight_command_failure(self):
        self.assert_retained(False)

    def test_invalid_expected_spec_is_rejected_before_rendering(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            args.spec = Path(folder) / "invalid.json"
            args.spec.write_text('{"expected":null}')
            with patch.object(handoff.subprocess, "run") as render:
                with self.assertRaises(ValueError):
                    handoff.render(args)
                render.assert_not_called()

    def test_failed_render_frees_its_revision_name(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            failure = subprocess.CalledProcessError(1, "raea render")
            with patch.object(handoff.subprocess, "run", side_effect=failure):
                with self.assertRaises(subprocess.CalledProcessError):
                    handoff.render(args)
            self.assertFalse(args.output.exists(), "an empty revision folder blocked the same name on the next run")


if __name__ == "__main__":
    unittest.main()
