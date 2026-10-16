#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用已有 records 目录里的 CSV / JSON 重新生成图文报告, 不重新分析音频。
用途: 报告样式更新后, 把旧报告重画一遍。

  python rebuild_report.py                 # 重画 records 下全部
  python rebuild_report.py records/xxx ...  # 只重画指定目录
"""
import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snore_detect as sd

FMT = "%Y-%m-%d %H:%M:%S"


def recluster(rows, cfg):
    """按当前配置重算归串, 让旧版(没有成串字段)的报告也能一致地重画。"""
    gap_th = cfg.get("cluster_gap_s", 5.0)
    cmin = cfg.get("cluster_min", 3)
    cid = 0
    for j, ev in enumerate(rows):
        if j == 0 or (ev["start_t"] - (rows[j - 1]["start_t"]
                                       + rows[j - 1]["duration_s"])) > gap_th:
            cid += 1
        ev["cluster"] = cid
    sizes = {}
    for ev in rows:
        sizes[ev["cluster"]] = sizes.get(ev["cluster"], 0) + 1
    for ev in rows:
        ev["cluster_size"] = sizes[ev["cluster"]]
        ev["likely_snore"] = ev["cluster_size"] >= cmin
    return cid


def load_dir(d):
    sp = glob.glob(os.path.join(d, "summary_*.json"))
    if not sp:
        return None
    with open(sp[0], encoding="utf-8") as f:
        summary = json.load(f)
    tag = os.path.basename(sp[0])[len("summary_"):-len(".json")]
    t0 = datetime.strptime(summary["start"], FMT)
    cfg = {**sd.DEFAULT_CFG, **(summary.get("cfg") or {})}
    summary["cfg"] = cfg

    rows = []
    ev = os.path.join(d, f"events_{tag}.csv")
    if os.path.exists(ev):
        with open(ev, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                st = datetime.strptime(r["start_time"], FMT)
                rows.append({
                    "start_t": (st - t0).total_seconds(),
                    "duration_s": float(r["duration_s"]),
                    "peak_dbfs": float(r["peak_dbfs"]),
                    "mean_low_ratio": float(r["low_ratio"]),
                    "clip": r.get("clip", ""),
                })

    segs = []
    lv = os.path.join(d, f"levels_{tag}.csv")
    if os.path.exists(lv):
        with open(lv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                st = datetime.strptime(r["time"], FMT)
                segs.append(((st - t0).total_seconds(),
                             float(r["dbfs"]), float(r["low_ratio"])))

    # 补齐新版报告需要的派生字段(旧 summary 里可能没有)
    cl_id = recluster(rows, cfg)
    likely = [e for e in rows if e["likely_snore"]]
    summary.setdefault("n_snore_events", len(rows))
    summary.setdefault("n_likely_snore", len(likely))
    summary.setdefault("n_clusters", cl_id)
    summary.setdefault("total_snore_s",
                       round(sum(e["duration_s"] for e in rows), 1))
    summary.setdefault("total_likely_snore_s",
                       round(sum(e["duration_s"] for e in likely), 1))
    return summary, rows, segs, tag


def main():
    ap = argparse.ArgumentParser(description="重画已有的打鼾图文报告")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("dirs", nargs="*", help="records 子目录(不给则全部)")
    ap.add_argument("--out", default=os.path.join(here, "records"))
    args = ap.parse_args()

    targets = args.dirs or sorted(
        d for d in glob.glob(os.path.join(args.out, "*")) if os.path.isdir(d))
    if not targets:
        print(f"{args.out} 下没有可重画的目录")
        return
    n = 0
    for d in targets:
        got = load_dir(d)
        if not got:
            print(f"跳过(无 summary): {d}")
            continue
        summary, rows, segs, tag = got
        p = sd.write_report(d, tag, summary, rows, segs, summary["cfg"])
        print(f"重画 {os.path.basename(d)} -> {os.path.basename(p)}  ({len(rows)} 事件)")
        n += 1
    print(f"完成 {n} 个")


if __name__ == "__main__":
    main()
