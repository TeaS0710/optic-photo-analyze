#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from shared_photo_study import ensure_dir, image_record, list_images, load_analysis_index, professional_reading, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract advanced photographic study metrics.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_dir = args.output_dir / "metrics"
    reports_dir = args.output_dir / "reports"
    ensure_dir(metrics_dir)
    ensure_dir(reports_dir)

    analysis_index = load_analysis_index(args.analysis_dir)
    image_paths = list_images(args.input_dir)
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{total}] {image_path.name}", flush=True)
        record = image_record(image_path, analysis_index.get(image_path.name))
        reading = professional_reading(record)
        write_json(metrics_dir / f"{image_path.stem}.json", record)

        report_lines = [
            f"# {image_path.name}",
            "",
            f"- Family: `{record['family']}`",
            f"- Title candidate: `{record['semantic'].get('short_title') or 'n/a'}`",
            f"- Orientation: `{record['metrics']['orientation']}`",
            f"- Aspect ratio: `{record['metrics']['aspect_ratio']}`",
            f"- Luminance mean: `{record['metrics']['luminance_mean']}`",
            f"- Dynamic range: `{record['metrics']['dynamic_range']}`",
            f"- Saturation mean: `{record['metrics']['saturation_mean']}`",
            f"- Colorfulness: `{record['metrics']['colorfulness']}`",
            f"- Warm balance: `{record['metrics']['warm_balance']}`",
            f"- Edge density: `{record['metrics']['edge_density']}`",
            f"- Sharpness: `{record['metrics']['sharpness']}`",
            f"- Symmetry score: `{record['metrics']['symmetry_score']}`",
            f"- Negative space ratio: `{record['metrics']['negative_space_ratio']}`",
            f"- Visual center: `({record['metrics']['visual_center_x']}, {record['metrics']['visual_center_y']})`",
            f"- Dominant colors: `{', '.join(record['metrics']['dominant_colors'])}`",
            "",
            "## Curatorial Summary",
            "",
            reading["curatorial_summary"],
            "",
            "## Optical / Material Reading",
            "",
            *[f"- {item}" for item in reading["observations"]],
            "",
            "## Semantic Alignment",
            "",
            f"- analysis_status: `{record['semantic']['analysis_status']}`",
            f"- subject_count: `{record['semantic']['subject_count']}`",
            f"- object_count: `{record['semantic']['object_count']}`",
            f"- text_present: `{record['semantic']['text_present']}`",
            f"- support_kind: `{record['semantic']['support_kind']}`",
            "",
            *[f"- {item}" for item in reading["semantic_notes"]],
            "",
            "## Objects / Tags",
            "",
            *[f"- {item}" for item in (record['semantic']['salient_objects'] or [])],
            "",
            "## Professional Notes",
            "",
            f"- Brightness family: `{record['metrics']['brightness_family']}`",
            f"- Saturation family: `{record['metrics']['saturation_family']}`",
            f"- Texture family: `{record['metrics']['texture_family']}`",
            f"- Chroma family: `{record['metrics']['chroma_family']}`",
            f"- Rule-of-thirds distance: `{record['metrics']['thirds_distance']}`",
            f"- Line balance: `{record['metrics']['line_balance']}`",
            f"- Diagonal energy: `{record['metrics']['diagonal_energy']}`",
            f"- Shadow clip ratio: `{record['metrics']['shadow_clip_ratio']}`",
            f"- Highlight clip ratio: `{record['metrics']['highlight_clip_ratio']}`",
        ]
        (reports_dir / f"{image_path.stem}.md").write_text("\n".join(report_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
