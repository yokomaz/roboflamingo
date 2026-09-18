import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TrainCustomScriptTest(unittest.TestCase):
    def test_uses_server_paths_and_training_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_python = Path(directory) / "python"
            fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
            fake_python.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{directory}:{environment['PATH']}"

            result = subprocess.run(
                ["bash", "scripts/train_custom.sh"],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

        arguments = result.stdout.splitlines()
        expected_pairs = {
            "--dataset": "/inspire/dataset/calvin/task-abcd-d/task_ABCD_D",
            "--vision-encoder": "/inspire/hdd/global_user/ky26319/models/clip_vit_l14/modelscope_clip_vit_large_patch14",
            "--language-model": "/inspire/hdd/global_user/ky26319/models/mpt_1b_dolly",
            "--backbone-checkpoint": "/inspire/hdd/global_user/ky26319/models/openflamingo_3b_vitl_mpt1b_langinstruct/checkpoint.pt",
            "--output": "./output_test_3",
            "--batch-size": "8",
            "--window-size": "32",
            "--steps": "10000",
            "--checkpoint-steps": "1000",
        }
        for option, value in expected_pairs.items():
            index = arguments.index(option)
            self.assertEqual(arguments[index + 1], value)
        self.assertIn("--local-files-only", arguments)

    def test_nohup_script_starts_foreground_script_and_records_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fake_nohup = directory / "nohup"
            capture = directory / "nohup_arguments.txt"
            output = directory / "output"
            fake_nohup.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$NOHUP_CAPTURE"\n'
            )
            fake_nohup.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{directory}:{environment['PATH']}"
            environment["NOHUP_CAPTURE"] = str(capture)
            environment["OUTPUT"] = str(output)

            subprocess.run(
                ["bash", "scripts/train_custom_nohup.sh"],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

            for _ in range(100):
                if capture.exists():
                    break
                time.sleep(0.01)
            arguments = capture.read_text().splitlines()
            self.assertEqual(arguments[:2], ["bash", str(ROOT / "scripts/train_custom.sh")])
            self.assertTrue((output / "train.pid").read_text().strip().isdigit())


if __name__ == "__main__":
    unittest.main()
