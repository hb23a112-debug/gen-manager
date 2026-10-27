"""gen_manager のテスト。  python test_gen_manager.py"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_manager as gm  # noqa: E402

try:
    from PIL import Image, PngImagePlugin
except ImportError:
    Image = None


def cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = gm.main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.root = self.dir / "lib"
        self.src = self.dir / "src"
        self.src.mkdir()
        self.m = gm.Manager(self.root)

    def tearDown(self):
        self.m.close()
        self._tmp.cleanup()

    def make(self, name, data=b"x"):
        p = self.src / name
        p.write_bytes(data)
        return p


class TestRecordAndOrganize(Base):
    def test_copy_organizes_by_layout_and_keeps_original(self):
        f = self.make("ComfyUI_00001_.PNG", b"img")
        rid = self.m.record(experiment="exp A", condition="lora=0.6", settings={"steps": 20},
                            files=[f], duration=12.5, created_at="2026-10-27T10:00:00")
        outs = self.m.get_outputs(rid)
        self.assertEqual(outs[0]["path"], f"2026-10-27/exp_A/{rid:04d}_lora=0.6_01.png")
        self.assertTrue(f.exists())                       # コピーなので元は残る
        self.assertTrue((self.root / outs[0]["path"]).is_file())
        self.assertEqual(outs[0]["size"], 3)

    def test_move_removes_original(self):
        f = self.make("a.png")
        self.m.record(files=[f], mode="move")
        self.assertFalse(f.exists())

    def test_in_place_does_not_touch_file(self):
        f = self.make("a.png")
        rid = self.m.record(files=[f], mode="none")
        self.assertTrue(f.exists())
        self.assertEqual(Path(self.m.get_outputs(rid)[0]["path"]), f.resolve())
        self.assertEqual(list(self.root.glob("*/")), [])  # 整理先フォルダは作られない

    def test_name_collision_gets_suffix(self):
        a, b = self.make("same.png", b"1"), self.make("same2.png", b"2")
        layout_m = gm.Manager(self.root, layout="{date}/fixed{ext}")
        r1 = layout_m.record(files=[a], created_at="2026-01-01T00:00:00")
        r2 = layout_m.record(files=[b], created_at="2026-01-01T00:00:00")
        p1, p2 = (layout_m.get_outputs(r)[0]["path"] for r in (r1, r2))
        self.assertNotEqual(p1, p2)
        self.assertEqual((self.root / p1).read_bytes(), b"1")
        self.assertEqual((self.root / p2).read_bytes(), b"2")
        layout_m.close()

    def test_bad_characters_in_names_are_sanitized(self):
        f = self.make("a.png")
        rid = self.m.record(experiment='a/b:c*"d', condition="CON", files=[f], created_at="2026-01-01T00:00:00")
        path = self.m.get_outputs(rid)[0]["path"]
        self.assertNotIn("..", path)
        self.assertEqual(path.split("/")[1], "a_b_c__d")
        self.assertIn("_CON", path)  # Windows予約名を避ける

    def test_unsafe_layout_rejected(self):
        for bad in ("../{date}{ext}", "/abs/{date}{ext}", "C:/x/{date}{ext}", "{nope}{ext}"):
            with self.assertRaises(gm.ManagerError, msg=bad):
                gm.Manager(self.root, layout=bad)

    def test_missing_file_marks_run_failed(self):
        rid = self.m.record(files=[self.src / "nothing.png"])
        r = self.m.list_runs(ids=[rid])[0]
        self.assertEqual(r["status"], "failed")
        self.assertIn("nothing.png", r["error"])


class TestRunContext(Base):
    def test_success_records_duration_and_outputs(self):
        f = self.make("o.png")
        with self.m.run(experiment="e", settings={"a": 1}) as r:
            r.add_output(f)
            r.set_metric("score", 0.5)
        row = self.m.list_runs(ids=[r.run_id])[0]
        self.assertEqual(row["status"], "ok")
        self.assertGreaterEqual(row["duration_sec"], 0)
        self.assertEqual(row["n_outputs"], 1)
        self.assertEqual(row["metrics"], {"score": 0.5})

    def test_exception_is_recorded_and_reraised(self):
        with self.assertRaises(RuntimeError):
            with self.m.run(experiment="e") as r:
                raise RuntimeError("CUDA out of memory")
        row = self.m.list_runs(ids=[r.run_id])[0]
        self.assertEqual(row["status"], "failed")
        self.assertIn("CUDA out of memory", row["error"])

    def test_suppress_swallows_exception(self):
        with self.m.run(suppress=True) as r:
            raise ValueError("boom")
        self.assertEqual(self.m.list_runs(ids=[r.run_id])[0]["status"], "failed")

    def test_keyboard_interrupt_recorded_as_failed(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.m.run() as r:
                raise KeyboardInterrupt()
        self.assertEqual(self.m.list_runs(ids=[r.run_id])[0]["status"], "failed")


class TestCompare(Base):
    def seed(self):
        for cond, steps, durs, fails, score in (
            ("A", 20, [10, 20, 30], 0, [0.5, 0.7, 0.6]),
            ("B", 30, [40, 50], 2, [0.9, 0.8]),
        ):
            for i, d in enumerate(durs):
                self.m.record(experiment="x", condition=cond, settings={"steps": steps, "cfg": 7, "extra": {"k": i}},
                              metrics={"score": score[i]}, duration=d)
            for _ in range(fails):
                self.m.record(experiment="x", condition=cond, settings={"steps": steps, "cfg": 7},
                              status="failed", duration=1, error="oom")

    def test_group_stats(self):
        self.seed()
        res = self.m.compare(self.m.list_runs(experiment="x"))
        g = {x["name"]: x for x in res["groups"]}
        self.assertEqual((g["A"]["n"], g["A"]["ok"], g["A"]["failed"]), (3, 3, 0))
        self.assertEqual(g["A"]["dur_mean"], 20)
        self.assertEqual(g["A"]["dur_median"], 20)
        self.assertEqual(g["B"]["success_rate"], 0.5)
        self.assertEqual(g["B"]["dur_mean"], 45)  # 失敗(1秒)は平均に含めない
        self.assertAlmostEqual(g["A"]["metrics"]["score"], 0.6)

    def test_diff_shows_only_differing_settings(self):
        self.seed()
        diff = self.m.compare(self.m.list_runs(experiment="x"))["diff"]
        self.assertIn("steps", diff)
        self.assertEqual(diff["steps"], {"A": "20", "B": "30"})
        self.assertNotIn("cfg", diff)           # 同じ値は出さない
        self.assertIn("extra.k", diff)          # ネストも展開される
        self.assertEqual(diff["extra.k"]["A"], "{0, 1, 2}")

    def test_compare_by_run(self):
        self.seed()
        rows = self.m.list_runs(ids=[1, 4])
        res = self.m.compare(rows, "run")
        self.assertEqual([g["name"] for g in res["groups"]], ["#1", "#4"])

    def test_empty_success_has_no_division_error(self):
        self.m.record(condition="Z", status="failed", error="x")
        g = self.m.compare(self.m.list_runs())["groups"][0]
        self.assertEqual(g["success_rate"], 0.0)
        self.assertIsNone(g["dur_mean"])


@unittest.skipIf(Image is None, "Pillow が無いため ingest を検証できません")
class TestIngest(Base):
    def png(self, name, prompt, color):
        p = self.src / name
        info = PngImagePlugin.PngInfo()
        if prompt is not None:
            info.add_text("prompt", json.dumps(prompt))
        Image.new("RGB", (4, 4), color).save(p, pnginfo=info)
        return p

    def test_ingest_reads_settings_and_is_idempotent(self):
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl/base.safetensors"}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat"}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad"}},
            "4": {"class_type": "KSampler", "inputs": {"seed": 5, "steps": 25, "cfg": 6.5, "sampler_name": "euler",
                                                        "scheduler": "normal", "denoise": 1,
                                                        "positive": ["2", 0], "negative": ["3", 0]}},
            "5": {"class_type": "LoraLoader", "inputs": {"lora_name": "x.safetensors", "strength_model": 0.7}},
        }
        self.png("a.png", prompt, "red")
        self.png("plain.png", None, "blue")
        code, out, _ = cli("ingest", self.src, "--root", self.root)
        self.assertEqual(code, 0)
        self.assertIn("登録 1件", out)
        row = self.m.list_runs()[0]
        self.assertEqual(row["condition"], "base")
        self.assertEqual(row["settings"]["steps"], 25)
        self.assertEqual(row["settings"]["positive"], "a cat")
        self.assertEqual(row["settings"]["loras"], ["x.safetensors@0.7"])
        code, out, _ = cli("ingest", self.src, "--root", self.root)
        self.assertIn("登録 0件 / 登録済みのためスキップ 1件", out)  # 二重登録しない
        self.assertEqual(len(self.m.list_runs()), 1)


class TestCLI(Base):
    def test_run_success_collects_new_files_from_watch_dir(self):
        watch = self.dir / "watch"
        watch.mkdir()
        (watch / "old.png").write_bytes(b"old")
        script = f"import pathlib; pathlib.Path(r'{watch}', 'new.png').write_bytes(b'new')"
        code, out, _ = cli("run", "-e", "t", "-c", "c1", "--set", "steps=20", "--set", "name=abc",
                           "--watch", watch, "--root", self.root, "--", sys.executable, "-c", script)
        self.assertEqual(code, 0)
        row = self.m.list_runs()[0]
        self.assertEqual((row["status"], row["n_outputs"]), ("ok", 1))  # old.png は含まれない
        self.assertEqual(row["settings"], {"steps": 20, "name": "abc"})  # 型が復元される

    def test_run_failure_propagates_exit_code(self):
        code, out, _ = cli("run", "-e", "t", "--root", self.root, "--", sys.executable, "-c", "import sys; sys.exit(3)")
        self.assertEqual(code, 3)
        row = self.m.list_runs()[0]
        self.assertEqual(row["status"], "failed")
        self.assertIn("終了コード 3", row["error"])

    def test_run_unknown_command(self):
        code, *_ = cli("run", "--root", self.root, "--", "no-such-command-xyz")
        self.assertEqual(code, 127)
        self.assertEqual(self.m.list_runs()[0]["status"], "failed")

    def test_require_output(self):
        watch = self.dir / "w"
        watch.mkdir()
        code, *_ = cli("run", "--watch", watch, "--require-output", "--root", self.root,
                       "--", sys.executable, "-c", "pass")
        self.assertEqual(self.m.list_runs()[0]["status"], "failed")

    def test_list_compare_export_check(self):
        for cond, d, st in (("A", 10, "ok"), ("A", 12, "ok"), ("B", 30, "failed")):
            f = self.make(f"{cond}{d}.png", cond.encode() + bytes([d]))
            cli("record", "-e", "x", "-c", cond, "--set", "steps=20" if cond == "A" else "steps=40",
                "--duration", d, "--status", st, "--error", "oom" if st == "failed" else "",
                f, "--root", self.root)
        code, out, _ = cli("list", "--root", self.root)
        self.assertIn("3件  成功 2 / 失敗 1", out)
        self.assertIn("失敗の内訳", out)
        code, out, _ = cli("list", "--status", "failed", "--root", self.root)
        self.assertIn("1件", out)
        code, out, _ = cli("compare", "-e", "x", "--csv", self.dir / "c.csv", "--root", self.root)
        self.assertIn("設定の違い", out)
        self.assertIn("steps", out)
        self.assertTrue((self.dir / "c.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
        code, out, _ = cli("export", "--csv", self.dir / "all.csv", "--root", self.root)
        self.assertIn("3件", out)
        self.assertEqual(cli("check", "--root", self.root)[0], 0)
        # 整理済みファイルを消すと check が検出する
        victim = next(self.root.rglob("*.png"))
        victim.unlink()
        code, out, _ = cli("check", "--root", self.root)
        self.assertEqual(code, 1)
        self.assertIn("見つかりません", out)

    def test_root_option_works_before_and_after_subcommand(self):
        other = self.dir / "other"
        self.assertEqual(cli("--root", other, "record", "-e", "p")[0], 0)
        self.assertEqual(cli("record", "-e", "q", "--root", other)[0], 0)
        with gm.Manager(other) as m2:
            self.assertEqual([r["experiment"] for r in m2.list_runs()], ["q", "p"])

    def test_bad_inputs_give_clean_errors(self):
        code, _, err = cli("record", "--set", "novalue", "--root", self.root)
        self.assertEqual(code, 2)
        self.assertIn("key=value", err)
        code, _, err = cli("show", "999", "--root", self.root)
        self.assertEqual(code, 2)
        code, _, err = cli("record", "--layout", "../x{ext}", "--root", self.root)
        self.assertEqual(code, 2)

    def test_table_alignment_with_japanese(self):
        t = gm.render_table(["名前", "x"], [["あいう", "1"], ["abc", "2"]]).splitlines()
        # 全角は幅2として数え、2列目の開始位置(表示幅)が行ごとに揃う
        col2 = lambda line, ch: gm._w(line[:line.index(ch)])
        self.assertEqual(col2(t[2], "1"), col2(t[3], "2"))
        self.assertEqual(col2(t[2], "1"), col2(t[0], "x"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
