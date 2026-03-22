#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared_photo_study import (
    build_contact_sheet,
    corpus_report_markdown,
    corpus_summary,
    ensure_dir,
    svg_histogram,
    svg_scatter,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build corpus figures, families and reports.")
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    figure_dir = args.output_dir / "figures"
    ensure_dir(figure_dir)

    records = []
    for path in sorted(args.metrics_dir.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    print(f"Loaded {len(records)} study records", flush=True)

    rows = []
    for record in records:
        row = {
            "file_name": record["file_name"],
            "family": record["family"],
            "orientation": record["metrics"]["orientation"],
            "aspect_ratio": record["metrics"]["aspect_ratio"],
            "luminance_mean": record["metrics"]["luminance_mean"],
            "dynamic_range": record["metrics"]["dynamic_range"],
            "saturation_mean": record["metrics"]["saturation_mean"],
            "colorfulness": record["metrics"]["colorfulness"],
            "warm_balance": record["metrics"]["warm_balance"],
            "edge_density": record["metrics"]["edge_density"],
            "sharpness": record["metrics"]["sharpness"],
            "entropy": record["metrics"]["entropy"],
            "symmetry_score": record["metrics"]["symmetry_score"],
            "negative_space_ratio": record["metrics"]["negative_space_ratio"],
            "subject_count": record["semantic"]["subject_count"],
            "object_count": record["semantic"]["object_count"],
            "text_present": record["semantic"]["text_present"],
        }
        rows.append(row)

    write_csv(args.output_dir / "metrics_summary.csv", rows)
    summary = corpus_summary(records)
    write_json(args.output_dir / "families.json", {
        "summary": summary,
        "records": [{"file_name": r["file_name"], "family": r["family"]} for r in records],
    })
    write_json(args.output_dir / "semantic_trends.json", {
        "top_objects": summary["top_objects"],
        "top_keywords": summary["top_keywords"],
        "support_kinds": summary["support_kinds"],
        "human_image_count": summary["human_image_count"],
        "text_image_count": summary["text_image_count"],
    })

    if records:
        svg_histogram([r["metrics"]["luminance_mean"] for r in records], "Brightness Distribution", "luminance_mean", figure_dir / "brightness_histogram.svg")
        svg_histogram([r["metrics"]["saturation_mean"] for r in records], "Saturation Distribution", "saturation_mean", figure_dir / "saturation_histogram.svg")
        svg_histogram([r["metrics"]["colorfulness"] for r in records], "Colorfulness Distribution", "colorfulness", figure_dir / "colorfulness_histogram.svg")
        svg_histogram([r["metrics"]["warm_balance"] for r in records], "Warm Balance Distribution", "warm_balance", figure_dir / "warm_balance_histogram.svg")
        svg_scatter(records, "sharpness", "dynamic_range", "Sharpness vs Dynamic Range", figure_dir / "sharpness_vs_dynamic_range.svg")
        svg_scatter(records, "saturation_mean", "edge_density", "Saturation vs Edge Density", figure_dir / "saturation_vs_edge_density.svg")
        svg_scatter(records, "visual_center_x", "visual_center_y", "Visual Center Map", figure_dir / "visual_center_map.svg")
        build_contact_sheet(records, figure_dir / "contact_sheet.png")

    (args.output_dir / "corpus_report.md").write_text(corpus_report_markdown(records, summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
