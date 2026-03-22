from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ANALYSIS_MAX_DIMENSION = 1600


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def stem_without_prefix(name: str) -> str:
    if "_" in name and name[:3].isdigit():
        return name.split("_", 1)[1]
    return name


def load_analysis_index(analysis_dir: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(analysis_dir.glob("*.analysis.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "file_name" in payload:
            index[payload["file_name"]] = payload
    return index


def rgb_to_hsv_np(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = arr.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc == 0, 0, delta / np.maximum(maxc, 1e-6))
    h = np.zeros_like(maxc)
    mask = delta > 1e-6
    rc = np.where(mask, (maxc - r) / np.maximum(delta, 1e-6), 0)
    gc = np.where(mask, (maxc - g) / np.maximum(delta, 1e-6), 0)
    bc = np.where(mask, (maxc - b) / np.maximum(delta, 1e-6), 0)
    h = np.where((r == maxc) & mask, bc - gc, h)
    h = np.where((g == maxc) & mask, 2.0 + rc - bc, h)
    h = np.where((b == maxc) & mask, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    return h, s, v


def dominant_colors(arr: np.ndarray, n_colors: int = 5) -> list[str]:
    image = Image.fromarray(arr.astype(np.uint8))
    image = ImageOps.contain(image, (256, 256))
    small = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=n_colors).convert("RGB")
    colors = Counter(tuple(pixel) for pixel in np.array(small).reshape(-1, 3))
    return [f"#{r:02x}{g:02x}{b:02x}" for (r, g, b), _ in colors.most_common(n_colors)]


def compute_metrics(image_path: Path) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    width_px, height_px = image.size
    study_image = image.copy()
    study_image.thumbnail((ANALYSIS_MAX_DIMENSION, ANALYSIS_MAX_DIMENSION), Image.Resampling.LANCZOS)
    arr = np.array(study_image, dtype=np.float32)
    h_px, w_px = arr.shape[:2]
    gray = np.dot(arr[..., :3], np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    h, s, v = rgb_to_hsv_np(arr)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    grad = np.sqrt(gx ** 2 + gy ** 2)
    edge_density = float((grad > np.percentile(grad, 85)).mean())
    sharpness = float(np.var(grad))
    entropy_counts, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
    entropy = float(-(entropy_counts[entropy_counts > 0] * np.log2(entropy_counts[entropy_counts > 0])).sum())

    luminance_mean = float(gray.mean())
    luminance_std = float(gray.std())
    dynamic_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    saturation_mean = float(s.mean())
    saturation_std = float(s.std())
    warm_balance = float((arr[..., 0].mean() - arr[..., 2].mean()) / 255.0)
    colorfulness = float(np.sqrt(np.var(arr[..., 0] - arr[..., 1]) + np.var(0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2])))
    shadow_clip_ratio = float((gray <= 8).mean())
    highlight_clip_ratio = float((gray >= 247).mean())

    weight = np.abs(gray - gray.mean()) + 1e-6
    yy, xx = np.indices(gray.shape)
    cx = float((xx * weight).sum() / weight.sum() / max(w_px - 1, 1))
    cy = float((yy * weight).sum() / weight.sum() / max(h_px - 1, 1))
    thirds = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]
    thirds_distance = min(math.dist((cx, cy), point) for point in thirds)
    horizontal_energy = float(np.mean(np.abs(gx)))
    vertical_energy = float(np.mean(np.abs(gy)))
    line_balance = float((horizontal_energy - vertical_energy) / max(horizontal_energy + vertical_energy, 1e-6))
    diagonal_energy = float(np.mean(np.abs(gray[1:, 1:] - gray[:-1, :-1])) + np.mean(np.abs(gray[1:, :-1] - gray[:-1, 1:])))

    left = gray[:, : max(1, w_px // 2)]
    right = np.fliplr(gray[:, w_px - left.shape[1]:])
    symmetry_score = 1.0 - float(np.mean(np.abs(left - right)) / 255.0)

    edge_pad_x = max(1, int(w_px * 0.12))
    edge_pad_y = max(1, int(h_px * 0.12))
    center = gray[edge_pad_y:h_px - edge_pad_y, edge_pad_x:w_px - edge_pad_x]
    edge_mask = np.ones_like(gray, dtype=bool)
    edge_mask[edge_pad_y:h_px - edge_pad_y, edge_pad_x:w_px - edge_pad_x] = False
    edges = gray[edge_mask]
    vignette_score = float((center.mean() - edges.mean()) / 255.0) if center.size and edges.size else 0.0
    negative_space_ratio = float((weight < np.percentile(weight, 30)).mean())

    aspect_ratio = float(width_px / max(height_px, 1))
    orientation = "portrait" if height_px > width_px else "landscape" if width_px > height_px else "square"
    brightness_family = "dark" if luminance_mean < 85 else "midtone" if luminance_mean < 170 else "bright"
    saturation_family = "muted" if saturation_mean < 0.18 else "balanced" if saturation_mean < 0.38 else "vivid"
    texture_family = "calm" if edge_density < 0.10 else "structured" if edge_density < 0.18 else "dense"
    chroma_family = "cold" if warm_balance < -0.03 else "warm" if warm_balance > 0.03 else "neutral"

    return {
        "width_px": width_px,
        "height_px": height_px,
        "study_width_px": w_px,
        "study_height_px": h_px,
        "aspect_ratio": round(aspect_ratio, 4),
        "orientation": orientation,
        "luminance_mean": round(luminance_mean, 4),
        "luminance_std": round(luminance_std, 4),
        "dynamic_range": round(dynamic_range, 4),
        "saturation_mean": round(saturation_mean, 4),
        "saturation_std": round(saturation_std, 4),
        "warm_balance": round(warm_balance, 4),
        "colorfulness": round(colorfulness, 4),
        "shadow_clip_ratio": round(shadow_clip_ratio, 6),
        "highlight_clip_ratio": round(highlight_clip_ratio, 6),
        "entropy": round(entropy, 4),
        "edge_density": round(edge_density, 6),
        "sharpness": round(sharpness, 4),
        "visual_center_x": round(cx, 4),
        "visual_center_y": round(cy, 4),
        "thirds_distance": round(thirds_distance, 4),
        "line_balance": round(line_balance, 4),
        "diagonal_energy": round(diagonal_energy, 4),
        "symmetry_score": round(symmetry_score, 4),
        "vignette_score": round(vignette_score, 4),
        "negative_space_ratio": round(negative_space_ratio, 4),
        "dominant_colors": dominant_colors(arr),
        "brightness_family": brightness_family,
        "saturation_family": saturation_family,
        "texture_family": texture_family,
        "chroma_family": chroma_family,
    }


def extract_semantic_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload or "error" in payload:
        return {
            "analysis_status": "missing_or_error",
            "subject_count": 0,
            "object_count": 0,
            "text_present": False,
            "keywords": [],
            "support_kind": None,
            "salient_objects": [],
            "subject_roles": [],
            "dominant_axes": [],
        }

    observe = payload.get("observe", {})
    text = payload.get("text", {})
    anchors = payload.get("anchors", {})
    writing = payload.get("writing", {})
    interpretation = payload.get("interpretation", payload.get("interpret", {}))
    critique = payload.get("critique", {})
    return {
        "analysis_status": "ok",
        "subject_count": len(observe.get("subjects", []) or []),
        "object_count": len(observe.get("salient_objects", []) or []),
        "text_present": bool((text.get("combined_text") or "").strip()),
        "keywords": writing.get("keywords", []) or [],
        "support_kind": payload.get("support", {}).get("support_kind"),
        "salient_objects": observe.get("salient_objects", []) or [],
        "subject_roles": [item.get("role", "") for item in observe.get("subjects", []) or []],
        "dominant_axes": anchors.get("dominant_axes", []) or [],
        "scene_summary": observe.get("scene_summary"),
        "core_reading": interpretation.get("core_reading"),
        "short_title": writing.get("short_title"),
        "keywords": writing.get("keywords", []) or [],
        "faithfulness_score": critique.get("faithfulness_score"),
        "overreach_risk_score": critique.get("overreach_risk_score"),
    }


def family_label(metrics: dict[str, Any], semantic: dict[str, Any]) -> str:
    text_tag = "text" if semantic.get("text_present") else "no-text"
    subject_tag = "human" if semantic.get("subject_count", 0) > 0 else "object"
    return f"{metrics['brightness_family']}-{metrics['saturation_family']}-{metrics['texture_family']}-{subject_tag}-{text_tag}"


def image_record(image_path: Path, payload: dict[str, Any] | None) -> dict[str, Any]:
    metrics = compute_metrics(image_path)
    semantic = extract_semantic_metrics(payload)
    family = family_label(metrics, semantic)
    return {
        "file_name": image_path.name,
        "image_path": str(image_path),
        "family": family,
        "metrics": metrics,
        "semantic": semantic,
        "run_metadata": payload.get("run_metadata", {}) if payload else {},
    }


def professional_reading(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record["metrics"]
    semantic = record["semantic"]
    observations: list[str] = []

    if metrics["brightness_family"] == "dark":
        observations.append("La matière tonale s'inscrit dans un registre sombre, avec une réserve de lumière relativement contenue.")
    elif metrics["brightness_family"] == "bright":
        observations.append("La scène est portée par une lumière globale élevée, avec une lecture visuelle très ouverte.")
    else:
        observations.append("La tonalité générale reste intermédiaire, ce qui conserve des marges de modelé dans les valeurs moyennes.")

    if metrics["dynamic_range"] > 120:
        observations.append("La dynamique tonale est ample, ce qui suggère une scène visuellement tendue entre zones denses et zones plus ouvertes.")
    else:
        observations.append("La dynamique tonale reste resserrée, avec un rendu plus compact et moins démonstratif.")

    if metrics["saturation_family"] == "muted":
        observations.append("La couleur travaille surtout en retenue: la palette paraît atténuée, presque documentaire.")
    elif metrics["saturation_family"] == "vivid":
        observations.append("La saturation occupe une place active dans l'image et participe clairement à son impact perceptif.")
    else:
        observations.append("La saturation demeure équilibrée, sans surcharge chromatique apparente.")

    if metrics["chroma_family"] == "warm":
        observations.append("L'équilibre colorimétrique tire vers le chaud, ce qui peut épaissir la présence des peaux, des poussières ou des matières minérales.")
    elif metrics["chroma_family"] == "cold":
        observations.append("La balance chromatique est plutôt froide, avec une sensation de distance et de retrait.")
    else:
        observations.append("L'équilibre chromatique reste relativement neutre.")

    if metrics["thirds_distance"] < 0.18:
        observations.append("Le centre d'intérêt se tient près d'un point fort de tiers, ce qui stabilise la lecture compositionnelle.")
    else:
        observations.append("Le centre d'intérêt s'écarte des tiers classiques, avec une organisation plus flottante ou moins académique.")

    if metrics["negative_space_ratio"] > 0.45:
        observations.append("Une part importante de l'image agit comme espace de réserve, ce qui laisse respirer les formes principales.")
    if metrics["symmetry_score"] > 0.82:
        observations.append("La répartition latérale est assez symétrique, ce qui renforce la frontalité ou la stabilité du cadre.")
    if metrics["vignette_score"] > 0.04:
        observations.append("Le centre apparaît plus lumineux que la périphérie, avec un léger effet de concentration optique.")
    if metrics["shadow_clip_ratio"] > 0.06 or metrics["highlight_clip_ratio"] > 0.06:
        observations.append("On observe un début de clipping tonal, donc une perte locale d'information dans les extrêmes.")

    semantic_notes: list[str] = []
    if semantic.get("scene_summary"):
        semantic_notes.append(f"Résumé de scène: {semantic['scene_summary']}")
    if semantic.get("core_reading"):
        semantic_notes.append(f"Lecture interprétative: {semantic['core_reading']}")
    if semantic.get("keywords"):
        semantic_notes.append(f"Mots-clés de sortie: {', '.join(semantic['keywords'][:8])}")
    if semantic.get("faithfulness_score") is not None:
        semantic_notes.append(
            f"Fiabilité sémantique estimée: {semantic['faithfulness_score']}/100, risque de débordement: {semantic.get('overreach_risk_score', 'n/a')}/100."
        )

    return {
        "curatorial_summary": " ".join(observations[:4]),
        "observations": observations,
        "semantic_notes": semantic_notes,
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def svg_histogram(values: list[float], title: str, xlabel: str, output_path: Path, bins: int = 12) -> None:
    width, height = 900, 420
    pad_left, pad_right, pad_top, pad_bottom = 70, 30, 50, 60
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    hist, edges = np.histogram(values, bins=bins)
    max_count = int(max(hist.max(), 1))
    bars = []
    for idx, count in enumerate(hist):
        x0 = pad_left + plot_w * idx / bins
        x1 = pad_left + plot_w * (idx + 1) / bins - 4
        bar_h = plot_h * (count / max_count)
        y = pad_top + plot_h - bar_h
        bars.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1-x0,1):.1f}" height="{bar_h:.1f}" fill="#3b6ea8"/>')
        label = f"{edges[idx]:.1f}"
        bars.append(f'<text x="{x0:.1f}" y="{height-20}" font-size="10">{label}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{pad_left}" y="28" font-size="22" font-family="Helvetica">{title}</text>
<line x1="{pad_left}" y1="{pad_top+plot_h}" x2="{pad_left+plot_w}" y2="{pad_top+plot_h}" stroke="black"/>
<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top+plot_h}" stroke="black"/>
{''.join(bars)}
<text x="{width/2-40:.1f}" y="{height-6}" font-size="12">{xlabel}</text>
</svg>'''
    output_path.write_text(svg, encoding="utf-8")


def svg_scatter(records: list[dict[str, Any]], x_key: str, y_key: str, title: str, output_path: Path) -> None:
    width, height = 900, 460
    pad_left, pad_right, pad_top, pad_bottom = 70, 40, 50, 60
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    xs = [record["metrics"][x_key] for record in records]
    ys = [record["metrics"][y_key] for record in records]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 1
    points = []
    for record in records:
        x = record["metrics"][x_key]
        y = record["metrics"][y_key]
        px = pad_left + (x - x_min) / (x_max - x_min) * plot_w
        py = pad_top + plot_h - (y - y_min) / (y_max - y_min) * plot_h
        points.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="#a63b6e"/>')
        points.append(f'<text x="{px+7:.1f}" y="{py-7:.1f}" font-size="10">{record["file_name"][:18]}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{pad_left}" y="28" font-size="22" font-family="Helvetica">{title}</text>
<line x1="{pad_left}" y1="{pad_top+plot_h}" x2="{pad_left+plot_w}" y2="{pad_top+plot_h}" stroke="black"/>
<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top+plot_h}" stroke="black"/>
{''.join(points)}
<text x="{width/2-80:.1f}" y="{height-8}" font-size="12">{x_key}</text>
<text x="8" y="{height/2:.1f}" font-size="12" transform="rotate(-90 8,{height/2:.1f})">{y_key}</text>
</svg>'''
    output_path.write_text(svg, encoding="utf-8")


def build_contact_sheet(records: list[dict[str, Any]], output_path: Path, thumb_size: tuple[int, int] = (220, 220)) -> None:
    cols = 3
    rows = max(1, math.ceil(len(records) / cols))
    margin = 20
    label_h = 50
    canvas = Image.new("RGB", (cols * (thumb_size[0] + margin) + margin, rows * (thumb_size[1] + label_h + margin) + margin), "white")
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(records):
        row, col = divmod(index, cols)
        x = margin + col * (thumb_size[0] + margin)
        y = margin + row * (thumb_size[1] + label_h + margin)
        image = Image.open(record["image_path"]).convert("RGB")
        thumb = ImageOps.contain(image, thumb_size)
        thumb_canvas = Image.new("RGB", thumb_size, "#f2f2f2")
        thumb_canvas.paste(thumb, ((thumb_size[0] - thumb.width) // 2, (thumb_size[1] - thumb.height) // 2))
        canvas.paste(thumb_canvas, (x, y))
        draw.text((x, y + thumb_size[1] + 8), record["file_name"][:28], fill="black")
        draw.text((x, y + thumb_size[1] + 24), record["family"][:28], fill="#555555")
    canvas.save(output_path)


def corpus_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    family_counter = Counter(record["family"] for record in records)
    brightness = [record["metrics"]["luminance_mean"] for record in records]
    saturation = [record["metrics"]["saturation_mean"] for record in records]
    sharpness = [record["metrics"]["sharpness"] for record in records]
    object_counter = Counter()
    keyword_counter = Counter()
    support_counter = Counter()
    for record in records:
        object_counter.update(record["semantic"].get("salient_objects", []))
        keyword_counter.update(record["semantic"].get("keywords", []))
        if record["semantic"].get("support_kind"):
            support_counter.update([record["semantic"]["support_kind"]])
    return {
        "image_count": len(records),
        "families": family_counter,
        "brightness_mean": round(float(np.mean(brightness)), 4) if brightness else None,
        "saturation_mean": round(float(np.mean(saturation)), 4) if saturation else None,
        "sharpness_mean": round(float(np.mean(sharpness)), 4) if sharpness else None,
        "mean_colorfulness": round(float(np.mean([record["metrics"]["colorfulness"] for record in records])), 4) if records else None,
        "human_image_count": sum(1 for record in records if record["semantic"].get("subject_count", 0) > 0),
        "text_image_count": sum(1 for record in records if record["semantic"].get("text_present")),
        "top_objects": object_counter.most_common(12),
        "top_keywords": keyword_counter.most_common(12),
        "support_kinds": support_counter.most_common(),
    }


def corpus_report_markdown(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    family_lines = "\n".join(f"- `{family}`: {count}" for family, count in summary["families"].most_common())
    strongest_contrast = sorted(records, key=lambda r: r["metrics"]["dynamic_range"], reverse=True)[:3]
    most_dense = sorted(records, key=lambda r: r["metrics"]["edge_density"], reverse=True)[:3]
    most_colorful = sorted(records, key=lambda r: r["metrics"]["colorfulness"], reverse=True)[:3]
    lines = [
        "# Corpus Photo Study",
        "",
        f"- Images étudiées : {summary['image_count']}",
        f"- Luminance moyenne : {summary['brightness_mean']}",
        f"- Saturation moyenne : {summary['saturation_mean']}",
        f"- Netteté moyenne : {summary['sharpness_mean']}",
        f"- Colorfulness moyen : {summary['mean_colorfulness']}",
        f"- Images avec présence humaine : {summary['human_image_count']}",
        f"- Images avec texte visible : {summary['text_image_count']}",
        "",
        "## Familles",
        "",
        family_lines or "- aucune famille",
        "",
        "## Objets saillants récurrents",
        "",
        *[f"- `{label}`: {count}" for label, count in summary["top_objects"]],
        "",
        "## Mots-clés récurrents",
        "",
        *[f"- `{label}`: {count}" for label, count in summary["top_keywords"]],
        "",
        "## Images à plus forte dynamique tonale",
        "",
        *[f"- `{record['file_name']}` — dynamic_range={record['metrics']['dynamic_range']}" for record in strongest_contrast],
        "",
        "## Images à texture visuelle dense",
        "",
        *[f"- `{record['file_name']}` — edge_density={record['metrics']['edge_density']}" for record in most_dense],
        "",
        "## Images à intensité chromatique forte",
        "",
        *[f"- `{record['file_name']}` — colorfulness={record['metrics']['colorfulness']}" for record in most_colorful],
        "",
        "## Lecture professionnelle",
        "",
        "Le corpus est ici lu à deux niveaux : matière photographique et lecture sémantique.",
        "Cela permet de distinguer non seulement ce que montrent les images, mais aussi comment elles le montrent.",
    ]
    return "\n".join(lines)
