#!/usr/bin/env python3
"""
gen_manager — 画像・動画・音声などの「生成結果」を管理するツール(標準ライブラリのみ)

できること
  1. 生成設定と出力ファイルを SQLite に記録する
  2. 出力ファイルを  日付/実験名/連番_条件名_xx.拡張子  のように自動で整理する
  3. 生成時間・成功/失敗を一覧にする
  4. 複数の検証条件(condition)ごとに、成功率・所要時間・設定の違いを比較する

使い方の例
  # 任意のコマンドを包んで実行(時間計測・失敗判定・出力ファイルの回収を自動化)
  python gen_manager.py run -e lora-test -c "lora=0.6" --set steps=20 --set cfg=7 \
        --watch ComfyUI/output -- python generate.py --lora 0.6

  # すでにある結果を手動で記録
  python gen_manager.py record -e lora-test -c "lora=0.8" --set steps=20 \
        --duration 41.5 out1.png out2.png

  # ComfyUI が PNG に埋め込んだ設定を読み取って、既存の出力をまとめて登録
  python gen_manager.py ingest ComfyUI/output -e comfy

  python gen_manager.py list --status failed
  python gen_manager.py compare -e lora-test
  python gen_manager.py compare --runs 3 5 8
  python gen_manager.py export --csv runs.csv

Python から使う場合
  from gen_manager import Manager
  mgr = Manager("gen_library")
  with mgr.run(experiment="lora-test", condition="lora=0.6", settings={"steps": 20}) as r:
      path = generate(...)          # 例外が出れば自動で「失敗」として記録される
      r.add_output(path)
      r.set_metric("score", 0.82)

元ファイルの扱い: 既定は「コピー」。--move で移動、--in-place で整理せず場所だけ記録。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

DEFAULT_ROOT = "gen_library"
DB_NAME = "gen_manager.db"
DEFAULT_LAYOUT = "{date}/{experiment}/{run:04d}_{condition}_{seq:02d}{ext}"
LAYOUT_FIELDS = "date, time, experiment, condition, run, seq, stem, ext"
SCHEMA_VERSION = 1

STATUS_OK, STATUS_FAILED, STATUS_RUNNING = "ok", "failed", "running"


class ManagerError(Exception):
    pass


# --------------------------------------------------------------------------
# 小さなユーティリティ
# --------------------------------------------------------------------------
_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL",
             *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def sanitize(value, maxlen: int = 60, fallback: str = "_") -> str:
    """ファイル名・フォルダ名として安全な文字列にする(Windowsの禁止文字・予約名も考慮)。"""
    s = _BAD_CHARS.sub("_", str(value)).strip(" .")
    s = re.sub(r"\s+", "_", s)
    if not s:
        return fallback
    if s.split(".")[0].upper() in _RESERVED:
        s = "_" + s
    return s[:maxlen]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def parse_value(text: str):
    """'20' -> 20, '0.6' -> 0.6, 'true' -> True, それ以外は文字列のまま。"""
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def parse_kv(pairs) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ManagerError(f"'key=value' の形式で指定してください: {p!r}")
        k, v = p.split("=", 1)
        k = k.strip()
        if not k:
            raise ManagerError(f"キーが空です: {p!r}")
        out[k] = parse_value(v)
    return out


def flatten(d: dict, prefix: str = "") -> dict:
    """ネストした設定を 'a.b' 形式の1階層にし、値は表示用の文字列にする。"""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return out


def fmt_duration(sec) -> str:
    if sec is None:
        return "-"
    sec = float(sec)
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{int(m)}m{s:04.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{s:04.1f}s"


# ---- 日本語を含む表の整形(全角=幅2で揃える) ----
def _w(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _clip(s: str, width: int) -> str:
    s = str(s).replace("\n", " ")
    if _w(s) <= width:
        return s
    out, acc = "", 0
    for c in s:
        cw = _w(c)
        if acc + cw > width - 1:
            break
        out, acc = out + c, acc + cw
    return out + "…"


def render_table(headers, rows, max_width: int = 40) -> str:
    cells = [[_clip(c, max_width) for c in r] for r in rows]
    heads = [_clip(h, max_width) for h in headers]
    widths = [max([_w(heads[i])] + [_w(r[i]) for r in cells]) for i in range(len(heads))]

    def line(r):
        return "  ".join(c + " " * (widths[i] - _w(c)) for i, c in enumerate(r)).rstrip()

    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(heads), sep] + [line(r) for r in cells])


def write_csv(path, headers, rows) -> None:
    # utf-8-sig: Windows の Excel でそのまま開いても文字化けしない
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


# --------------------------------------------------------------------------
# 記録の本体
# --------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    experiment   TEXT NOT NULL DEFAULT '',
    condition    TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL,
    duration_sec REAL,
    error        TEXT NOT NULL DEFAULT '',
    settings     TEXT NOT NULL DEFAULT '{}',
    metrics      TEXT NOT NULL DEFAULT '{}',
    note         TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS outputs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    path          TEXT NOT NULL,      -- root からの相対(整理済み) or 絶対パス(--in-place)
    original_path TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    size          INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_exp ON runs(experiment, condition);
CREATE INDEX IF NOT EXISTS idx_out_run ON outputs(run_id);
CREATE INDEX IF NOT EXISTS idx_out_sha ON outputs(sha256);
"""


class Manager:
    def __init__(self, root=DEFAULT_ROOT, layout: str = DEFAULT_LAYOUT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.layout = layout
        self.db = sqlite3.connect(self.root / DB_NAME)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        ver = self.db.execute("PRAGMA user_version").fetchone()[0]
        if ver > SCHEMA_VERSION:
            raise ManagerError(f"DB のバージョン({ver})がこのツールより新しいです。ツールを更新してください")
        self.db.executescript(SCHEMA)
        self.db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.db.commit()
        self._check_layout()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- 公開API: 実行を包む ----
    def run(self, experiment="", condition="", settings=None, metrics=None, note="",
            mode="copy", suppress=False, created_at=None):
        return RunContext(self, experiment, condition, settings, metrics, note, mode, suppress, created_at)

    # ---- 公開API: 結果を後から1件記録 ----
    def record(self, *, experiment="", condition="", settings=None, metrics=None, note="",
               files=(), status=STATUS_OK, duration=None, error="", mode="copy",
               created_at=None, known_sha=None) -> int:
        if status not in (STATUS_OK, STATUS_FAILED):
            raise ManagerError(f"status は ok / failed のどちらかです: {status!r}")
        run_id = self._begin(experiment, condition, settings, metrics, note, created_at)
        self._finish(run_id, status, duration, error, files, mode, known_sha=known_sha)
        return run_id

    # ---- 内部: 開始と終了 ----
    def _begin(self, experiment, condition, settings, metrics, note, created_at=None) -> int:
        cur = self.db.execute(
            "INSERT INTO runs (created_at, experiment, condition, status, settings, metrics, note)"
            " VALUES (?,?,?,?,?,?,?)",
            (created_at or now_iso(), experiment or "", condition or "", STATUS_RUNNING,
             jdump(settings or {}), jdump(metrics or {}), note or ""),
        )
        self.db.commit()
        return cur.lastrowid

    def _finish(self, run_id, status, duration, error, files, mode, settings=None, metrics=None,
                known_sha=None):
        errors = []
        for seq, f in enumerate(files, start=1):
            try:
                self._attach(run_id, Path(f), seq, mode, (known_sha or {}).get(str(f)))
            except (OSError, ManagerError) as e:
                errors.append(f"{f}: {e}")
        if errors and status == STATUS_OK:
            status = STATUS_FAILED
        if errors:
            error = "; ".join(filter(None, [error, *errors]))
        sets, vals = ["status=?", "duration_sec=?", "error=?"], [status, duration, error]
        if settings is not None:
            sets.append("settings=?"); vals.append(jdump(settings))
        if metrics is not None:
            sets.append("metrics=?"); vals.append(jdump(metrics))
        self.db.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id=?", (*vals, run_id))
        self.db.commit()
        return status

    # ---- 整理: 保存先の決定とコピー/移動 ----
    def _check_layout(self):
        dummy = self._render_layout({"date": "2000-01-01", "time": "000000", "experiment": "e",
                                     "condition": "c", "run": 1, "seq": 1, "stem": "s", "ext": ".png"})
        return dummy

    def _render_layout(self, fields: dict) -> Path:
        try:
            rel = self.layout.format(**fields)
        except (KeyError, IndexError, ValueError) as e:
            raise ManagerError(f"--layout が不正です({e!r})。使えるフィールド: {LAYOUT_FIELDS}")
        rel = rel.replace("\\", "/")
        parts = rel.split("/")
        if (rel.startswith("/") or re.match(r"^[A-Za-z]:", rel)
                or any(p in ("", "..", ".") for p in parts)):
            raise ManagerError(f"--layout の結果が不正です(絶対パス・空要素・.. は使えません): {rel!r}")
        return Path(*parts)

    def _attach(self, run_id: int, src: Path, seq: int, mode: str, sha=None):
        if mode not in ("copy", "move", "none"):
            raise ManagerError(f"mode は copy / move / none: {mode!r}")
        if not src.is_file():
            raise ManagerError("ファイルが見つかりません")
        sha = sha or sha256_of(src)
        size = src.stat().st_size

        if mode == "none":
            stored = str(src.resolve())
        else:
            run = self.db.execute("SELECT created_at, experiment, condition FROM runs WHERE id=?",
                                  (run_id,)).fetchone()
            ts = run["created_at"]
            fields = {
                "date": ts[:10],
                "time": ts[11:19].replace(":", ""),
                "experiment": sanitize(run["experiment"], fallback="misc"),
                "condition": sanitize(run["condition"], fallback="base"),
                "run": run_id,
                "seq": seq,
                "stem": sanitize(src.stem, 80, "file"),
                "ext": src.suffix.lower(),
            }
            dest = self.root / self._render_layout(fields)
            root_r = self.root.resolve()
            try:
                dest.resolve().relative_to(root_r)
            except ValueError:
                raise ManagerError(f"保存先が root の外になります: {dest}")
            if dest.exists() and dest.resolve() == src.resolve():
                pass  # すでにその場所にある
            else:
                dest = self._unique(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if mode == "move":
                    shutil.move(str(src), str(dest))
                else:
                    shutil.copy2(src, dest)
            stored = dest.resolve().relative_to(root_r).as_posix()

        self.db.execute(
            "INSERT INTO outputs (run_id, path, original_path, sha256, size) VALUES (?,?,?,?,?)",
            (run_id, stored, str(src.resolve()), sha, size),
        )

    @staticmethod
    def _unique(p: Path) -> Path:
        if not p.exists():
            return p
        for n in range(2, 100000):
            cand = p.with_name(f"{p.stem}-{n}{p.suffix}")
            if not cand.exists():
                return cand
        raise ManagerError(f"重複しない名前を作れませんでした: {p}")

    def resolve_output(self, stored: str) -> Path:
        return self.root / stored  # 絶対パスならそのまま絶対パスになる

    # ---- 参照 ----
    def list_runs(self, experiment=None, condition=None, status=None, since=None, until=None,
                  ids=None, limit=None):
        q = ("SELECT r.*, (SELECT COUNT(*) FROM outputs o WHERE o.run_id=r.id) AS n_outputs"
             " FROM runs r WHERE 1=1")
        p = []
        if experiment is not None:
            q += " AND r.experiment=?"; p.append(experiment)
        if condition is not None:
            q += " AND r.condition=?"; p.append(condition)
        if status:
            q += " AND r.status=?"; p.append(status)
        if since:
            q += " AND r.created_at>=?"; p.append(since)
        if until:
            q += " AND r.created_at<=?"; p.append(until + "T23:59:59" if len(until) == 10 else until)
        if ids:
            q += f" AND r.id IN ({','.join('?' * len(ids))})"; p.extend(ids)
        q += " ORDER BY r.id DESC"
        if limit:
            q += " LIMIT ?"; p.append(int(limit))
        rows = [self._row(r) for r in self.db.execute(q, p).fetchall()]
        return rows

    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        for k in ("settings", "metrics"):
            try:
                d[k] = json.loads(d[k]) or {}
            except ValueError:
                d[k] = {}
        return d

    def get_outputs(self, run_id: int):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM outputs WHERE run_id=? ORDER BY id", (run_id,))]

    def has_sha(self, sha: str) -> bool:
        return self.db.execute("SELECT 1 FROM outputs WHERE sha256=? LIMIT 1", (sha,)).fetchone() is not None

    # ---- 集計・比較 ----
    @staticmethod
    def summarize(rows) -> dict:
        ok = [r for r in rows if r["status"] == STATUS_OK]
        failed = [r for r in rows if r["status"] == STATUS_FAILED]
        durs = [r["duration_sec"] for r in ok if r["duration_sec"] is not None]
        done = len(ok) + len(failed)
        return {
            "n": len(rows), "ok": len(ok), "failed": len(failed),
            "running": len(rows) - done,
            "success_rate": (len(ok) / done) if done else None,
            "dur_mean": statistics.fmean(durs) if durs else None,
            "dur_median": statistics.median(durs) if durs else None,
            "dur_min": min(durs) if durs else None,
            "dur_max": max(durs) if durs else None,
            "outputs": sum(r["n_outputs"] for r in rows),
        }

    def compare(self, rows, by: str = "condition") -> dict:
        """rows を by(condition / experiment / run)でグループ化して比較する。

        所要時間は「成功した実行」だけで集計する(失敗は途中で止まるため短く出て、平均を歪める)。
        """
        keyfn = {
            "condition": lambda r: r["condition"] or "(なし)",
            "experiment": lambda r: r["experiment"] or "(なし)",
            "run": lambda r: f"#{r['id']}",
        }.get(by)
        if keyfn is None:
            raise ManagerError("--by は condition / experiment / run のいずれかです")
        groups: "OrderedDict[str, list]" = OrderedDict()
        for r in sorted(rows, key=lambda r: r["id"]):
            groups.setdefault(keyfn(r), []).append(r)

        out_groups = []
        for name, rs in groups.items():
            s = self.summarize(rs)
            s["name"] = name
            metric_vals = {}
            for r in rs:
                for k, v in r["metrics"].items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        metric_vals.setdefault(k, []).append(v)
            s["metrics"] = {k: statistics.fmean(v) for k, v in metric_vals.items()}
            out_groups.append(s)

        # 設定の違い: グループ間で値が異なる(または片方にしか無い)キーだけを出す
        per_group = []
        for name, rs in groups.items():
            vals = {}
            for r in rs:
                for k, v in flatten(r["settings"]).items():
                    vals.setdefault(k, [])
                    if v not in vals[k]:
                        vals[k].append(v)
            per_group.append((name, vals))
        all_keys = sorted({k for _, v in per_group for k in v})
        diff = OrderedDict()
        for k in all_keys:
            cells = {}
            for name, vals in per_group:
                v = vals.get(k)
                cells[name] = "-" if not v else (v[0] if len(v) == 1 else "{" + ", ".join(v) + "}")
            if len(set(cells.values())) > 1:
                diff[k] = cells
        return {"by": by, "groups": out_groups, "diff": diff}

    # ---- 整合性チェック ----
    def check(self, verify_hash: bool = False):
        problems = []
        for o in self.db.execute("SELECT * FROM outputs ORDER BY run_id, id"):
            p = self.resolve_output(o["path"])
            if not p.is_file():
                problems.append((o["run_id"], o["path"], "ファイルが見つかりません"))
            elif p.stat().st_size != o["size"]:
                problems.append((o["run_id"], o["path"], "サイズが記録と違います"))
            elif verify_hash and sha256_of(p) != o["sha256"]:
                problems.append((o["run_id"], o["path"], "ハッシュが記録と違います"))
        return problems


class RunContext:
    """with mgr.run(...) as r:  の r。例外が出たら失敗として記録する(Ctrl+C も含む)。"""

    def __init__(self, mgr, experiment, condition, settings, metrics, note, mode, suppress, created_at):
        self.mgr, self.mode, self.suppress = mgr, mode, suppress
        self.settings = dict(settings or {})
        self.metrics = dict(metrics or {})
        self.files = []
        self.error = ""
        self._args = (experiment, condition, note, created_at)
        self.run_id = None
        self.status = None

    def __enter__(self):
        exp, cond, note, created = self._args
        self.run_id = self.mgr._begin(exp, cond, self.settings, self.metrics, note, created)
        self._t0 = time.perf_counter()
        return self

    def add_output(self, path):
        self.files.append(path)

    def set_setting(self, key, value):
        self.settings[key] = value

    def set_metric(self, key, value):
        self.metrics[key] = value

    def fail(self, message: str):
        """例外を出さずに失敗として扱いたいとき(例: 外部コマンドの終了コード)。"""
        self.error = message

    def __exit__(self, exc_type, exc, tb):
        duration = time.perf_counter() - self._t0
        status = STATUS_OK
        if exc_type is not None:
            status = STATUS_FAILED
            self.error = f"{exc_type.__name__}: {exc}"
        elif self.error:
            status = STATUS_FAILED
        try:
            self.status = self.mgr._finish(self.run_id, status, duration, self.error, self.files,
                                           self.mode, settings=self.settings, metrics=self.metrics)
        except Exception as fin_err:  # 記録の失敗で元の例外を隠さない
            print(f"[gen_manager] 記録の確定に失敗: {fin_err}", file=sys.stderr)
        return bool(self.suppress and exc_type is not None and issubclass(exc_type, Exception))


# --------------------------------------------------------------------------
# ComfyUI の PNG に埋め込まれた設定の読み取り(ingest 用・Pillow があれば)
# --------------------------------------------------------------------------
def extract_comfy_settings(prompt: dict) -> dict:
    s, loras = {}, []
    sampler_inputs = None
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        ct, inp = node.get("class_type", ""), node.get("inputs") or {}
        if sampler_inputs is None and ("KSampler" in ct or "SamplerCustom" in ct):
            sampler_inputs = inp
            for k_out, k_in in (("sampler", "sampler_name"), ("scheduler", "scheduler"),
                                ("steps", "steps"), ("cfg", "cfg"), ("denoise", "denoise")):
                if inp.get(k_in) is not None and not isinstance(inp.get(k_in), list):
                    s[k_out] = inp[k_in]
            seed = inp.get("seed", inp.get("noise_seed"))
            if seed is not None and not isinstance(seed, list):
                s["seed"] = seed
        if "CheckpointLoader" in ct and "model" not in s and inp.get("ckpt_name"):
            s["model"] = inp["ckpt_name"]
        if "Lora" in ct or "LoRA" in ct:
            if isinstance(inp.get("lora_name"), str):
                loras.append(f"{inp['lora_name']}@{inp.get('strength_model', inp.get('strength', ''))}")
    if loras:
        s["loras"] = loras
    if sampler_inputs:
        for key in ("positive", "negative"):
            link = sampler_inputs.get(key)
            if isinstance(link, list) and link:
                text = (prompt.get(str(link[0])) or {}).get("inputs", {}).get("text")
                if isinstance(text, str):
                    s[key] = text
    return s


def read_png_settings(path: Path):
    try:
        from PIL import Image
    except ImportError:
        raise ManagerError("ingest には Pillow が必要です (pip install pillow)")
    try:
        with Image.open(path) as img:
            raw = img.info.get("prompt")
        return extract_comfy_settings(json.loads(raw)) if raw else None
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _snapshot(directory: Path, skip: Path):
    snap = {}
    skip = skip.resolve()
    for p in directory.rglob("*"):
        try:
            if p.is_file() and skip not in p.resolve().parents:
                st = p.stat()
                snap[str(p)] = (st.st_mtime_ns, st.st_size)
        except OSError:
            continue
    return snap


def _collect_settings(args) -> dict:
    settings = {}
    if getattr(args, "settings_file", None):
        settings.update(json.loads(Path(args.settings_file).read_text(encoding="utf-8")))
    if getattr(args, "settings", None):
        settings.update(json.loads(args.settings))
    settings.update(parse_kv(getattr(args, "set", None)))
    return settings


def _print_run_line(mgr, run_id):
    r = mgr.list_runs(ids=[run_id])[0]
    mark = "OK " if r["status"] == STATUS_OK else "FAIL"
    print(f"#{run_id} {mark} {fmt_duration(r['duration_sec'])} 出力{r['n_outputs']}件"
          + (f"  エラー: {r['error']}" if r["error"] else ""))


def cmd_run(mgr, args) -> int:
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise ManagerError("実行するコマンドを '--' の後ろに指定してください")
    watch = Path(args.watch) if args.watch else None
    before = _snapshot(watch, mgr.root) if watch else {}
    t_start = time.time()
    exit_code = 0
    with mgr.run(experiment=args.experiment, condition=args.condition,
                 settings=_collect_settings(args), metrics=parse_kv(args.metric),
                 note=args.note, mode=args.mode, suppress=True) as r:
        try:
            exit_code = subprocess.run(cmd).returncode
        except FileNotFoundError:
            exit_code = 127
            r.fail(f"コマンドが見つかりません: {cmd[0]}")
        if exit_code not in (0, 127):
            r.fail(f"終了コード {exit_code}")
        for f in args.file or []:
            r.add_output(f)
        if watch:
            after = _snapshot(watch, mgr.root)
            new = sorted(p for p, sig in after.items() if before.get(p) != sig)
            for p in new:
                r.add_output(p)
            if args.require_output and not new and not r.error and exit_code == 0:
                r.fail("出力ファイルが生成されませんでした")
    _print_run_line(mgr, r.run_id)
    return exit_code


def cmd_record(mgr, args) -> int:
    run_id = mgr.record(experiment=args.experiment, condition=args.condition,
                        settings=_collect_settings(args), metrics=parse_kv(args.metric),
                        note=args.note, files=args.files, status=args.status,
                        duration=args.duration, error=args.error, mode=args.mode)
    _print_run_line(mgr, run_id)
    return 0


def cmd_ingest(mgr, args) -> int:
    base = Path(args.directory)
    if not base.is_dir():
        raise ManagerError(f"フォルダがありません: {base}")
    skip = mgr.root.resolve()
    added = skipped_dup = skipped_nometa = 0
    for p in sorted(base.rglob("*.png")):
        if skip in p.resolve().parents:
            continue
        sha = sha256_of(p)
        if mgr.has_sha(sha):
            skipped_dup += 1
            continue
        settings = read_png_settings(p)
        if settings is None:
            skipped_nometa += 1
            if not args.include_no_meta:
                continue
            settings = {}
        st = p.stat()
        cond = args.condition or (Path(settings["model"]).stem if settings.get("model") else "")
        mgr.record(experiment=args.experiment, condition=cond, settings=settings, files=[p],
                   mode=args.mode, created_at=datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%dT%H:%M:%S"),
                   known_sha={str(p): sha})
        added += 1
    print(f"登録 {added}件 / 登録済みのためスキップ {skipped_dup}件 / "
          f"設定なし{'(登録)' if args.include_no_meta else 'のためスキップ'} {skipped_nometa}件")
    return 0


def _settings_brief(settings: dict, keys, limit=4) -> str:
    flat = flatten(settings)
    ks = keys or list(flat)[:limit]
    return " ".join(f"{k}={flat[k]}" for k in ks if k in flat)


def cmd_list(mgr, args) -> int:
    rows = mgr.list_runs(args.experiment, args.condition, args.status, args.since, args.until,
                         limit=args.limit)
    if not rows:
        print("該当する記録がありません")
        return 0
    keys = [k.strip() for k in args.keys.split(",")] if args.keys else None
    label = {STATUS_OK: "OK", STATUS_FAILED: "FAIL", STATUS_RUNNING: "RUN?"}
    table = [[r["id"], r["created_at"].replace("T", " "), r["experiment"], r["condition"],
              label.get(r["status"], r["status"]), fmt_duration(r["duration_sec"]), r["n_outputs"],
              _settings_brief(r["settings"], keys)] for r in rows]
    print(render_table(["ID", "日時", "実験", "条件", "状態", "時間", "出力", "設定"], table))
    s = mgr.summarize(rows)
    rate = "-" if s["success_rate"] is None else f"{s['success_rate'] * 100:.0f}%"
    print(f"\n{s['n']}件  成功 {s['ok']} / 失敗 {s['failed']}"
          + (f" / 未完了 {s['running']}" if s["running"] else "")
          + f"  成功率 {rate}  平均時間(成功のみ) {fmt_duration(s['dur_mean'])}")
    if s["running"]:
        print("※ RUN? は実行中、または途中でプロセスが強制終了された記録です")
    failed = [r for r in rows if r["status"] == STATUS_FAILED and r["error"]]
    if failed and not args.status:
        print("\n失敗の内訳:")
        for r in failed[:10]:
            print(f"  #{r['id']}  {_clip(r['error'], 100)}")
    return 0


def cmd_show(mgr, args) -> int:
    rows = mgr.list_runs(ids=[args.id])
    if not rows:
        raise ManagerError(f"#{args.id} は見つかりません")
    r = rows[0]
    print(f"#{r['id']}  {r['created_at']}  状態: {r['status']}  時間: {fmt_duration(r['duration_sec'])}")
    print(f"実験: {r['experiment'] or '-'}   条件: {r['condition'] or '-'}")
    if r["error"]:
        print(f"エラー: {r['error']}")
    if r["note"]:
        print(f"メモ: {r['note']}")
    print("設定:")
    for k, v in flatten(r["settings"]).items():
        print(f"  {k}: {v}")
    if r["metrics"]:
        print("指標: " + ", ".join(f"{k}={v}" for k, v in r["metrics"].items()))
    print("出力:")
    for o in mgr.get_outputs(r["id"]) or []:
        print(f"  {o['path']}  ({o['size']:,} bytes)")
    return 0


def _pct(x):
    return "-" if x is None else f"{x * 100:.0f}%"


def cmd_compare(mgr, args) -> int:
    rows = mgr.list_runs(args.experiment, None, args.status, args.since, args.until,
                         ids=args.runs or None)
    if not rows:
        print("比較対象の記録がありません")
        return 0
    by = args.by or ("run" if args.runs else "condition")
    res = mgr.compare(rows, by)
    metric_keys = sorted({k for g in res["groups"] for k in g["metrics"]})
    headers = [{"condition": "条件", "experiment": "実験", "run": "実行"}[by],
               "件数", "成功", "失敗", "成功率", "平均時間", "中央値", "最短", "最長", "出力数"]
    headers += [f"平均{k}" for k in metric_keys]
    table = []
    for g in res["groups"]:
        row = [g["name"], g["n"], g["ok"], g["failed"], _pct(g["success_rate"]),
               fmt_duration(g["dur_mean"]), fmt_duration(g["dur_median"]),
               fmt_duration(g["dur_min"]), fmt_duration(g["dur_max"]), g["outputs"]]
        row += [f"{g['metrics'][k]:.4g}" if k in g["metrics"] else "-" for k in metric_keys]
        table.append(row)
    print("■ 結果の比較(時間は成功した実行のみで集計)")
    print(render_table(headers, table))
    if args.csv:
        write_csv(args.csv, headers, table)
        print(f"\nCSVを書き出しました: {args.csv}")

    if len(res["groups"]) >= 2:
        print("\n■ 設定の違い(グループ間で値が異なる項目のみ。{a, b} はグループ内で値がばらついている)")
        if res["diff"]:
            names = [g["name"] for g in res["groups"]]
            print(render_table(["設定項目"] + names,
                               [[k] + [cells[n] for n in names] for k, cells in res["diff"].items()],
                               max_width=28))
        else:
            print("(違いはありません)")
    return 0


def cmd_export(mgr, args) -> int:
    rows = mgr.list_runs(args.experiment, args.condition, args.status, args.since, args.until)
    rows.sort(key=lambda r: r["id"])
    headers = ["id", "created_at", "experiment", "condition", "status", "duration_sec", "error",
               "n_outputs", "settings", "metrics", "outputs", "note"]
    data = [[r["id"], r["created_at"], r["experiment"], r["condition"], r["status"],
             "" if r["duration_sec"] is None else round(r["duration_sec"], 3), r["error"],
             r["n_outputs"], jdump(r["settings"]), jdump(r["metrics"]),
             ";".join(o["path"] for o in mgr.get_outputs(r["id"])), r["note"]] for r in rows]
    write_csv(args.csv, headers, data)
    print(f"{len(data)}件を書き出しました: {args.csv}")
    return 0


def cmd_check(mgr, args) -> int:
    problems = mgr.check(args.hash)
    if not problems:
        print("問題は見つかりませんでした")
        return 0
    for run_id, path, msg in problems:
        print(f"#{run_id}  {path}  {msg}")
    print(f"\n{len(problems)}件の問題があります")
    return 1


def build_parser() -> argparse.ArgumentParser:
    root_help = f"管理フォルダ(既定: ./{DEFAULT_ROOT}、環境変数 GEN_MANAGER_ROOT)"
    layout_help = f"保存先テンプレート(既定: {DEFAULT_LAYOUT})。使える項目: {LAYOUT_FIELDS}"
    # --root / --layout はサブコマンドの前後どちらに書いても効くようにする。
    # サブコマンド側の既定値は SUPPRESS にして、前に書いた値を上書きしない。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=argparse.SUPPRESS, help=root_help)
    common.add_argument("--layout", default=argparse.SUPPRESS, help=layout_help)

    runopts = argparse.ArgumentParser(add_help=False)
    runopts.add_argument("-e", "--experiment", default="", help="実験名(フォルダ分けにも使う)")
    runopts.add_argument("-c", "--condition", default="", help="検証条件のラベル(比較の単位)")
    runopts.add_argument("--set", action="append", metavar="KEY=VALUE", help="生成設定(複数指定可)")
    runopts.add_argument("--settings", metavar="JSON", help="生成設定をJSON文字列で")
    runopts.add_argument("--settings-file", metavar="FILE", help="生成設定のJSONファイル")
    runopts.add_argument("--metric", action="append", metavar="KEY=NUM", help="評価値(例: score=0.8)")
    runopts.add_argument("--note", default="")
    g = runopts.add_mutually_exclusive_group()
    g.add_argument("--move", dest="mode", action="store_const", const="move",
                   help="出力ファイルを整理先へ移動(既定はコピー)")
    g.add_argument("--in-place", dest="mode", action="store_const", const="none",
                   help="ファイルは動かさず、場所だけ記録")
    runopts.set_defaults(mode="copy")

    filt = argparse.ArgumentParser(add_help=False)
    filt.add_argument("-e", "--experiment", default=None)
    filt.add_argument("--status", choices=[STATUS_OK, STATUS_FAILED, STATUS_RUNNING])
    filt.add_argument("--since", metavar="YYYY-MM-DD")
    filt.add_argument("--until", metavar="YYYY-MM-DD")

    p = argparse.ArgumentParser(prog="gen_manager", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.environ.get("GEN_MANAGER_ROOT", DEFAULT_ROOT), help=root_help)
    p.add_argument("--layout", default=DEFAULT_LAYOUT, help=layout_help)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("run", parents=[common, runopts], help="コマンドを実行し、時間・成否・出力を記録")
    s.add_argument("--watch", metavar="DIR", help="このフォルダに新しくできた/更新されたファイルを出力として回収")
    s.add_argument("-f", "--file", action="append", help="出力ファイルを明示(複数指定可)")
    s.add_argument("--require-output", action="store_true", help="--watch で何も増えなければ失敗扱い")
    s.add_argument("cmd", nargs=argparse.REMAINDER, help="-- の後ろに実行するコマンド")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("record", parents=[common, runopts], help="結果を後から1件記録")
    s.add_argument("files", nargs="*", help="出力ファイル")
    s.add_argument("--status", choices=[STATUS_OK, STATUS_FAILED], default=STATUS_OK)
    s.add_argument("--duration", type=float, metavar="SEC")
    s.add_argument("--error", default="")
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("ingest", parents=[common], help="ComfyUI の PNG から設定を読み取って一括登録")
    s.add_argument("directory")
    s.add_argument("-e", "--experiment", default="comfyui")
    s.add_argument("-c", "--condition", default="", help="省略時はモデル名")
    s.add_argument("--include-no-meta", action="store_true", help="設定が埋め込まれていないPNGも登録")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--move", dest="mode", action="store_const", const="move")
    g.add_argument("--in-place", dest="mode", action="store_const", const="none")
    s.set_defaults(func=cmd_ingest, mode="copy")

    s = sub.add_parser("list", parents=[common, filt], help="実行の一覧(時間・成否)")
    s.add_argument("-c", "--condition", default=None)
    s.add_argument("--limit", type=int, default=50)
    s.add_argument("--keys", help="設定欄に出す項目(カンマ区切り。例: steps,cfg)")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("show", parents=[common], help="1件の詳細")
    s.add_argument("id", type=int)
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("compare", parents=[common, filt], help="条件ごとの成功率・時間・設定差を比較")
    s.add_argument("--runs", type=int, nargs="+", metavar="ID", help="特定の実行同士を比較")
    s.add_argument("--by", choices=["condition", "experiment", "run"])
    s.add_argument("--csv", metavar="FILE", help="比較表をCSVにも書き出す")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("export", parents=[common, filt], help="全記録をCSVに書き出す")
    s.add_argument("-c", "--condition", default=None)
    s.add_argument("--csv", required=True, metavar="FILE")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("check", parents=[common], help="記録したファイルが残っているか確認")
    s.add_argument("--hash", action="store_true", help="ハッシュも照合(遅い)")
    s.set_defaults(func=cmd_check)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with Manager(args.root, args.layout) as mgr:
            return args.func(mgr, args)
    except ManagerError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as e:  # 不正なJSON、ファイル操作の失敗など
        print(f"エラー: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
