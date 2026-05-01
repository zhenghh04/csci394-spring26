#!/usr/bin/env python3
"""Build the four-system scaling plot + summary PDF for the MPI pi assignment."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ANS = Path(__file__).resolve().parent
LINE_RE = re.compile(r"procs=(\d+)\s+samples=(\d+)\s+hits=(\d+)\s+pi[≈=]([\d.]+)\s+time_s=([\d.]+)")

SYSTEMS = [
    ("laptop",  "MacBook Pro M1 Max (10 cores)",                     "tab:blue",   "o"),
    ("crux",    "ALCF Crux (1 node, 2x AMD EPYC 7742, 128 cores)",   "tab:orange", "s"),
    ("polaris", "ALCF Polaris (1 node, AMD EPYC Milan, 32 cores + 4xA100)", "tab:green", "^"),
    ("aurora",  "ALCF Aurora (1 node, Intel Xeon Max SR, 104 cores + 6xPVC)", "tab:red",  "d"),
]


def parse(path: Path) -> list[tuple[int, float]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        m = LINE_RE.search(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(5))))
    rows.sort()
    return rows


def main() -> None:
    fig_strong, ax_strong = plt.subplots(figsize=(8, 6))
    fig_speed, ax_speed = plt.subplots(figsize=(8, 6))

    found_any = False
    summaries: list[tuple[str, list[tuple[int, float]]]] = []
    for short, label, color, marker in SYSTEMS:
        data = parse(ANS / short / "results.txt")
        summaries.append((label, data))
        if not data:
            print(f"WARN: no data for {short}")
            continue
        found_any = True
        nprocs = [n for n, _ in data]
        times  = [t for _, t in data]
        t1 = times[0]
        speedups = [t1 / t for t in times]

        ax_strong.loglog(nprocs, times, marker=marker, color=color, label=label, lw=1.6)
        ax_speed.loglog(nprocs, speedups, marker=marker, color=color, label=label, lw=1.6)

    missing = [label for label, data in summaries if not data]
    if not found_any:
        sys.exit("No scaling data found in any answers/<system>/results.txt")

    # Ideal speedup line based on the union of x values.
    all_n = sorted({n for _, data in summaries for n, _ in data})
    if all_n:
        ax_speed.loglog(all_n, all_n, "k--", lw=1.0, label="ideal (linear)")

    for ax, ylabel, title in [
        (ax_strong, "max wall-time per run [s]", "MPI pi Monte-Carlo strong scaling (samples = 1e8)"),
        (ax_speed,  "speedup vs n=1 on the same system", "MPI pi Monte-Carlo speedup (samples = 1e8)"),
    ]:
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(fontsize=9)

    fig_strong.tight_layout()
    fig_speed.tight_layout()

    fig_strong.savefig(ANS / "scaling_time.png", dpi=150)
    fig_speed.savefig(ANS / "scaling_speedup.png", dpi=150)

    pdf_path = ANS / "mpi_pi_scaling_report.pdf"
    with PdfPages(pdf_path) as pdf:
        # Cover with both plots
        pdf.savefig(fig_strong)
        pdf.savefig(fig_speed)
        # Add a per-system data table as a separate page.
        fig_table, ax_table = plt.subplots(figsize=(8.5, 11))
        ax_table.axis("off")
        text_lines = [
            "MPI pi (mpi4py) strong-scaling assignment",
            "samples per run = 100,000,000",
            "",
        ]
        if missing:
            text_lines.append("Systems with no live data this run:")
            for m in missing:
                text_lines.append(f"  - {m}")
            text_lines.append("  (Aurora ClearML task stayed 'queued / k8s pending scheduler'")
            text_lines.append("   on the gpu_hack queue; the slurm-glue agent was alive but")
            text_lines.append("   never claimed the task. Live Aurora numbers were not captured.)")
            text_lines.append("")
        for label, data in summaries:
            text_lines.append(label)
            text_lines.append("-" * len(label))
            if not data:
                text_lines.append("  (no data)")
            else:
                text_lines.append(f"  {'procs':>6}  {'time_s':>10}  {'speedup':>8}")
                t1 = data[0][1]
                for n, t in data:
                    text_lines.append(f"  {n:>6d}  {t:>10.4f}  {t1/t:>8.2f}")
            text_lines.append("")
        ax_table.text(0.05, 0.97, "\n".join(text_lines), va="top", ha="left",
                      family="monospace", fontsize=9)
        pdf.savefig(fig_table)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {ANS / 'scaling_time.png'}")
    print(f"Wrote {ANS / 'scaling_speedup.png'}")


if __name__ == "__main__":
    main()
