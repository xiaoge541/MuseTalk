"""CPU tests for the Modal adapter.

Executed here: gate logic, call-id order, ffmpeg on synthetic color/sine
media, static reads of the MuseTalk entry, weight plan, import refusal.

Not executed: Modal, GPU, weights, MuseTalk inference, a model MP4.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from modal_musetalk.call_store import CallStore  # noqa: E402
from modal_musetalk.facts import README_MODEL_SENTENCE, THIRD_PARTY, VRAM_EVIDENCE  # noqa: E402
from modal_musetalk.media_commands import audio_to_16k_mono, remux_original_audio, still_container_from_png  # noqa: E402
from modal_musetalk.gate import (  # noqa: E402
    REQUIRED_STATEMENT,
    SCHEMA,
    evaluate_gate,
    import_requires_gate,
)
from modal_musetalk.input_spec import (  # noqa: E402
    EXPECTED_INFERENCE_FRAMES_FROM_DURATION,
    STILL_CONTAINER_FRAMES,
    expected_inference_frames,
    validate_original_wav,
    validate_portrait_png,
)
from modal_musetalk.job import inference_argv, run_job  # noqa: E402
from modal_musetalk.orchestrate import AlreadySubmitted, WaitTimeout, recover_once, submit_once  # noqa: E402
from modal_musetalk.spec import (  # noqa: E402
    GPU,
    RETRIES,
    SCALEDOWN_WINDOW_S,
    modal_image_ignore,
)
from modal_musetalk.untested import UNTESTED  # noqa: E402
from modal_musetalk.weights import known_hf_bytes, plan  # noqa: E402


def _valid_attestation(**overrides) -> dict:
    payload = {
        "schema": SCHEMA,
        "statement": REQUIRED_STATEMENT,
        "human_verified_at": "2026-11-18T00:00:00+08:00",
        "available_credit_usd": "recorded-by-a-person",
        "payment_requirement": "person read the billing page and wrote what it requires",
        "explicit_saved_cash_spend_cap_usd": 0,
        "cash_cap_is_saved_effective": True,
        "cash_cap_is_not_docs_default_formula": True,
        "gpu_eligibility_verified": True,
        "expired_window_is_not_authorization": True,
        "single_task_only": True,
    }
    payload.update(overrides)
    return payload


class GateTests(unittest.TestCase):
    def test_default_is_closed_without_reading_a_balance(self):
        decision = evaluate_gate({})
        self.assertFalse(decision.opened)
        self.assertIn("spend gate closed", decision.reason)

    def test_expired_window_does_not_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attestation.json"
            path.write_text(
                json.dumps(_valid_attestation(human_verified_at="2026-11-17T01:41:00+08:00")),
                encoding="utf-8",
            )
            decision = evaluate_gate({"MUSETALK_SPEND_ATTESTATION": str(path)})
        self.assertFalse(decision.opened)

    def test_docs_default_formula_does_not_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attestation.json"
            path.write_text(
                json.dumps(_valid_attestation(cash_cap_is_not_docs_default_formula=False)),
                encoding="utf-8",
            )
            decision = evaluate_gate({"MUSETALK_SPEND_ATTESTATION": str(path)})
        self.assertFalse(decision.opened)

    def test_unverified_gpu_does_not_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attestation.json"
            path.write_text(
                json.dumps(_valid_attestation(gpu_eligibility_verified=False)),
                encoding="utf-8",
            )
            decision = evaluate_gate({"MUSETALK_SPEND_ATTESTATION": str(path)})
        self.assertFalse(decision.opened)

    def test_credit_string_alone_does_not_open_when_cap_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attestation.json"
            payload = _valid_attestation()
            payload["available_credit_usd"] = 30
            payload["explicit_saved_cash_spend_cap_usd"] = None
            path.write_text(json.dumps(payload), encoding="utf-8")
            decision = evaluate_gate({"MUSETALK_SPEND_ATTESTATION": str(path)})
        self.assertFalse(decision.opened)

    def test_complete_external_attestation_opens_the_decision_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attestation.json"
            path.write_text(json.dumps(_valid_attestation()), encoding="utf-8")
            decision = evaluate_gate({"MUSETALK_SPEND_ATTESTATION": str(path)})
        self.assertTrue(decision.opened)
        self.assertFalse((REPO / "attestation.json").exists())

    def test_remote_marker_does_not_open_the_spend_decision(self):
        self.assertTrue(import_requires_gate({}))
        self.assertFalse(import_requires_gate({"MODAL_IS_REMOTE": "1"}))
        decision = evaluate_gate({"MODAL_IS_REMOTE": "1"})
        self.assertFalse(decision.opened)
        cli = (REPO / "modal_musetalk" / "cli.py").read_text(encoding="utf-8")
        self.assertNotIn("MODAL_IS_REMOTE", cli)


class OrchestrateTests(unittest.TestCase):
    def test_persists_call_id_before_get_and_does_not_resubmit(self):
        store = CallStore(Path(tempfile.mkdtemp()) / "calls")
        state = {"spawns": 0, "gets": 0}

        class Call:
            object_id = "fc-once"

            def get(self, timeout=None):
                state["gets"] += 1
                record = store.read("job-001")
                self_id = record["call_id"] if record else ""
                if self_id != "fc-once":
                    raise AssertionError("call id was not persisted before get")
                return {"mp4": b"not-a-real-musetalk-mp4"}

        class Ops:
            def ensure_function(self):
                return "function"

            def spawn(self, function, spawn_args):
                state["spawns"] += 1
                return Call()

        result = submit_once(
            ops=Ops(),
            store=store,
            job_id="job-001",
            spawn_args=(b"png", b"wav", "job-001"),
            wait_s=5,
            gpu=GPU,
        )
        self.assertEqual(result.call_id, "fc-once")
        self.assertEqual(state["spawns"], 1)
        self.assertEqual(state["gets"], 1)
        with self.assertRaises(AlreadySubmitted):
            submit_once(
                ops=Ops(),
                store=store,
                job_id="job-001",
                spawn_args=(b"png", b"wav", "job-001"),
                wait_s=5,
                gpu=GPU,
            )
        self.assertEqual(state["spawns"], 1)

    def test_timeout_keeps_the_same_id_and_recover_does_not_spawn(self):
        store = CallStore(Path(tempfile.mkdtemp()) / "calls")
        state = {"spawns": 0, "from_id": 0}

        class Call:
            def __init__(self, object_id):
                self.object_id = object_id

            def get(self, timeout=None):
                raise TimeoutError("still running")

        class Ops:
            def ensure_function(self):
                return "function"

            def spawn(self, function, spawn_args):
                state["spawns"] += 1
                return Call("fc-timeout")

            def from_id(self, call_id):
                state["from_id"] += 1
                self.recovered = call_id
                return Call(call_id)

        ops = Ops()
        with self.assertRaises(WaitTimeout) as caught:
            submit_once(
                ops=ops,
                store=store,
                job_id="job-002",
                spawn_args=(b"a", b"b", "job-002"),
                wait_s=1,
                gpu=GPU,
            )
        self.assertEqual(caught.exception.call_id, "fc-timeout")
        self.assertEqual(store.read("job-002")["status"], "waiting_recovery")
        with self.assertRaises(TimeoutError):
            recover_once(ops=ops, call_id="fc-timeout", wait_s=0)
        self.assertEqual(state["spawns"], 1)
        self.assertEqual(state["from_id"], 1)
        self.assertEqual(ops.recovered, "fc-timeout")

    def test_empty_call_id_does_not_wait_or_spawn_twice(self):
        store = CallStore(Path(tempfile.mkdtemp()) / "calls")
        state = {"spawns": 0, "gets": 0}

        class Call:
            object_id = ""

            def get(self, timeout=None):
                state["gets"] += 1

        class Ops:
            def ensure_function(self):
                return "function"

            def spawn(self, function, spawn_args):
                state["spawns"] += 1
                return Call()

        with self.assertRaises(RuntimeError):
            submit_once(
                ops=Ops(),
                store=store,
                job_id="job-003",
                spawn_args=(b"a", b"b", "job-003"),
                wait_s=1,
                gpu=GPU,
            )
        self.assertEqual(state["spawns"], 1)
        self.assertEqual(state["gets"], 0)
        self.assertIsNone(store.read("job-003"))


class FfmpegAndJobTests(unittest.TestCase):
    def _synthetic(self, root: Path) -> tuple[Path, Path, str, str]:
        portrait = root / "portrait.png"
        audio = root / "audio.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=gray:s=1086x1448",
                "-frames:v", "1", str(portrait),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "sine=frequency=440:sample_rate=48000:duration=3.377708",
                "-ac", "2", "-c:a", "pcm_s16le", str(audio),
            ],
            check=True,
            capture_output=True,
        )
        return portrait, audio, _sha(portrait), _sha(audio)

    def test_commands_do_not_overwrite_sources(self):
        with self.assertRaises(ValueError):
            audio_to_16k_mono(Path("/tmp/audio.wav"), Path("/tmp/audio.wav"))
        cmd = audio_to_16k_mono(Path("/tmp/audio_original.wav"), Path("/tmp/audio_16k_mono.wav"))
        self.assertEqual(cmd[cmd.index("-ar") + 1], "16000")
        self.assertEqual(cmd[cmd.index("-ac") + 1], "1")
        self.assertNotIn("input_still_container_NOT_DYNAMIC.mp4", cmd)
        still = still_container_from_png(Path("/tmp/portrait.png"), Path("/tmp/still.mp4"))
        self.assertIn("-an", still)
        self.assertIn("yuv420p", still)

    def test_synthetic_preprocess_and_fake_inference_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            portrait, audio, portrait_sha, audio_sha = self._synthetic(root)
            validate_portrait_png(portrait)
            validate_original_wav(audio)
            seen = []

            def runner(cmd, cwd=None):
                seen.append(list(cmd))
                if len(cmd) >= 3 and cmd[2] == "scripts.inference":
                    result_dir = Path(cmd[cmd.index("--result_dir") + 1])
                    out = result_dir / "v15"
                    out.mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64",
                            "-frames:v", "4", "-r", "25", "-pix_fmt", "yuv420p",
                            str(out / "result.mp4"),
                        ],
                        check=True,
                        capture_output=True,
                    )
                    return
                subprocess.run(cmd, check=True, capture_output=True)

            runner.accepts_cwd = True
            prepared = {"called": False}

            def weights(_models):
                prepared["called"] = True

            result = run_job(
                repo_root=root,
                job_dir=root / "job",
                portrait_png=portrait.read_bytes(),
                audio_wav=audio.read_bytes(),
                job_id="job-cpu",
                runner=runner,
                weight_preparer=weights,
                measured_on_gpu=False,
            )
            self.assertTrue(prepared["called"])
            self.assertFalse(result["usage"]["measured_on_gpu"])
            self.assertIsNone(result["usage"]["cold_start_s"])
            self.assertIsNone(result["usage"]["modal_dashboard_usd"])
            self.assertIsNone(result["usage"]["inference_subprocess_s"])
            config = (root / "job" / "job.yaml").read_text(encoding="utf-8")
            self.assertIn("portrait.png", config)
            self.assertIn("audio_16k_mono.wav", config)
            self.assertNotIn("NOT_DYNAMIC", config)
            self.assertNotIn(".mp4", config.split("audio_path")[0])
            inference_cmds = [cmd for cmd in seen if "scripts.inference" in cmd]
            self.assertEqual(len(inference_cmds), 1)
            self.assertIn("--version", inference_cmds[0])
            self.assertIn("v15", inference_cmds[0])
            self.assertIn("--use_float16", inference_cmds[0])
            self.assertNotIn("app.py", inference_cmds[0])
            audio_cmds = [cmd for cmd in seen if "-ar" in cmd]
            self.assertTrue(audio_cmds)
            self.assertEqual(audio_cmds[0][audio_cmds[0].index("-ar") + 1], "16000")
            streams = result["ffprobe"]["streams"]
            kinds = {stream.get("codec_type") for stream in streams}
            self.assertIn("video", kinds)
            self.assertIn("audio", kinds)
            self.assertEqual(_sha(portrait), portrait_sha)
            self.assertEqual(_sha(audio), audio_sha)
            derived = root / "job" / "audio_16k_mono.wav"
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(derived),
            ], text=True))
            stream = probe["streams"][0]
            self.assertEqual(stream["sample_rate"], "16000")
            self.assertEqual(stream["channels"], 1)
            self.assertEqual(stream["codec_name"], "pcm_s16le")

    def test_rejects_still_container_and_mono_16k_as_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            still = root / "still.mp4"
            mono = root / "mono.wav"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=gray:s=1086x1448",
                    "-t", "0.2", "-r", "25", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                    str(still),
                ],
                check=True, capture_output=True,
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=0.2",
                    "-ac", "1", "-c:a", "pcm_s16le", str(mono),
                ],
                check=True, capture_output=True,
            )
            with self.assertRaises(ValueError):
                validate_portrait_png(still)
            with self.assertRaises(ValueError):
                validate_original_wav(mono)
            remux = remux_original_audio(still, mono, root / "out.mp4")
            self.assertIn("-c:v", remux)
            self.assertEqual(remux[-1], str(root / "out.mp4"))


class EntryAndWeightTests(unittest.TestCase):
    def test_inference_entry_accepts_one_image_and_cleans_up_safely(self):
        text = (REPO / "scripts" / "inference.py").read_text(encoding="utf-8")
        self.assertIn('input_img_list = [video_path]', text)
        self.assertIn("save_dir_full = None", text)
        self.assertIn("if save_dir_full is not None:", text)
        self.assertIn('default="v15"', text)
        self.assertNotIn("ensure_wav", text)
        audio = (REPO / "musetalk" / "utils" / "audio_processor.py").read_text(encoding="utf-8")
        self.assertIn("librosa.load(wav_path, sr=16000)", audio)
        app = (REPO / "app.py").read_text(encoding="utf-8")
        self.assertIn("input img folder", app)
        utils = (REPO / "musetalk" / "utils" / "utils.py").read_text(encoding="utf-8")
        self.assertIn("return 'image'", utils)
        self.assertIn("'.png'", utils)

    def test_frame_formula_uses_audio_duration_not_the_still_container(self):
        self.assertEqual(EXPECTED_INFERENCE_FRAMES_FROM_DURATION, 84)
        self.assertEqual(expected_inference_frames(3.377708, 25), 84)
        self.assertNotEqual(EXPECTED_INFERENCE_FRAMES_FROM_DURATION, STILL_CONTAINER_FRAMES)

    def test_weight_plan_is_pinned_and_has_no_volume(self):
        payload = plan()
        self.assertIsNone(payload["volume"])
        self.assertEqual(payload["known_hf_bytes"], known_hf_bytes())
        self.assertGreater(payload["known_hf_bytes"], 4_000_000_000)
        revisions = {item["repo_id"]: item["revision"] for item in payload["hf"]}
        self.assertEqual(len(revisions["TMElyralab/MuseTalk"]), 40)
        self.assertNotIn("ByteDance/LatentSync", revisions)
        skipped = payload["not_downloaded"][0]
        self.assertEqual(skipped["hf_license_field"], "openrail++")
        self.assertIn("not", skipped["why"])
        argv = inference_argv(REPO, Path("/tmp/job.yaml"), Path("/tmp/results"))
        self.assertEqual(argv[2], "scripts.inference")
        self.assertIn("v15", argv)

    def test_image_ignore_skips_people_and_media(self):
        self.assertTrue(modal_image_ignore(Path("assets/demo/man/man.png")))
        self.assertTrue(modal_image_ignore(Path("data/audio/eng.wav")))
        self.assertTrue(modal_image_ignore(Path("models/musetalkV15/unet.pth")))
        self.assertFalse(modal_image_ignore(Path("scripts/inference.py")))
        self.assertFalse(modal_image_ignore(Path("modal_app.py")))

    def test_modal_app_is_gated_and_does_not_declare_a_volume(self):
        text = (REPO / "modal_app.py").read_text(encoding="utf-8")
        gate_at = text.index("require_open_gate_or_exit()")
        modal_at = text.index("import modal")
        self.assertLess(gate_at, modal_at)
        self.assertNotIn("modal.Volume", text)
        self.assertNotIn("modal.Secret", text)
        self.assertNotIn("web_endpoint", text)
        self.assertNotIn("create_if_missing", text)
        self.assertIn("scaledown_window=SCALEDOWN_WINDOW_S", text)
        self.assertIn("retries=RETRIES", text)
        self.assertEqual(SCALEDOWN_WINDOW_S, 2)
        self.assertEqual(RETRIES, 0)
        backend = (REPO / "modal_musetalk" / "modal_backend.py").read_text(encoding="utf-8")
        self.assertIn("FunctionCall.from_id", backend)
        self.assertIn("NotFoundError", backend)
        self.assertNotIn(".spawn", backend.split("def spawn", 1)[0])

    def test_closed_cli_does_not_import_modal(self):
        env = os.environ.copy()
        env.pop("MUSETALK_SPEND_ATTESTATION", None)
        env.pop("MUSETALK_MODAL_APP_IMPORT", None)
        proc = subprocess.run(
            [sys.executable, "-m", "modal_musetalk", "submit",
             "--portrait", "PORTRAIT.png", "--audio", "AUDIO.wav",
             "--output", "./out/result.mp4", "--job-id", "job-001"],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("spend gate closed", proc.stderr)
        self.assertNotIn("token", proc.stderr.lower())
        guard = subprocess.run(
            [sys.executable, "-c", "import modal_app"],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(guard.returncode, 0)
        self.assertIn("spend gate closed", guard.stderr)
        self.assertNotIn("Traceback", guard.stderr)

    def test_license_notes_do_not_treat_unknowns_as_a_grant(self):
        blob = json.dumps({"third_party": THIRD_PARTY, "vram": VRAM_EVIDENCE})
        self.assertNotIn("NOASSERTION", blob)
        self.assertIn("openrail++", blob)
        self.assertIn("Unknown", blob)
        self.assertIn("not a clearance", README_MODEL_SENTENCE.lower() + " " + blob.lower())
        joined = "\n".join(UNTESTED)
        for item in (
            "MuseTalk 1.5 inference",
            "an MP4 produced by MuseTalk",
            "Modal dashboard billed seconds or USD",
            "modal image build",
        ):
            self.assertIn(item, joined)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
