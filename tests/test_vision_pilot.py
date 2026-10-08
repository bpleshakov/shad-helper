import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.vision_pilot import main


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
class VisionPilotTests(unittest.TestCase):
    def test_requested_time_is_replaced_by_decoded_recording_relative_pts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "offset.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                "testsrc2=size=320x180:rate=20:duration=2", "-c:v", "mpeg4",
                "-output_ts_offset", "5", str(media),
            ], check=True)
            gold = root / "reference.json"
            gold.write_text(json.dumps({"cases": [{"id": "offset",
                "source_filename": media.name, "inspected_timestamps_seconds": [0.51]}]}))
            out = root / "pilot"
            real_run = subprocess.run

            def run(command, **kwargs):
                if command[:3] == [sys.executable, "-m", "shad_helper.vision_worker"]:
                    return subprocess.CompletedProcess(command, 0)
                return real_run(command, **kwargs)

            argv = ["vision_pilot.py", "--media-dir", str(root), "--model", "test/model",
                    "--revision", "test", "--out", str(out), "--gold", str(gold)]
            with (patch.object(sys, "argv", argv), patch.dict(os.environ),
                  patch("scripts.vision_pilot.subprocess.run", side_effect=run),
                  contextlib.redirect_stdout(io.StringIO())):
                main()
            item = json.loads((out / "request.json").read_text())["items"][0]
            self.assertEqual(item["requested_timestamp"], 0.51)
            self.assertAlmostEqual(item["timestamp"], 0.55, places=6)
            self.assertTrue(Path(item["image"]).is_file())
