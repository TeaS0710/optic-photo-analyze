#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from html import escape
from pathlib import Path

from PIL import Image, ImageOps

from shared_photo_study import ensure_dir, professional_reading


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static website from photo corpus study artifacts.")
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def copy_figures(source_dir: Path, target_dir: Path) -> None:
    ensure_dir(target_dir)
    for path in sorted(source_dir.glob("*")):
        if path.is_file():
            shutil.copy2(path, target_dir / path.name)


def build_thumbnail(image_path: Path, output_path: Path, size: tuple[int, int] = (720, 720)) -> None:
    ensure_dir(output_path.parent)
    image = Image.open(image_path).convert("RGB")
    thumb = ImageOps.contain(image, size)
    canvas = Image.new("RGB", size, "#efe7db")
    canvas.paste(thumb, ((size[0] - thumb.width) // 2, (size[1] - thumb.height) // 2))
    canvas.save(output_path, quality=90)


def metric_chip(label: str, value: str) -> str:
    return f'<div class="metric-chip"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'


def figure_card(name: str, title: str) -> str:
    return f'''
    <article class="figure-card">
      <div class="figure-frame">
        <img src="assets/figures/{escape(name)}" alt="{escape(title)}">
      </div>
      <h3>{escape(title)}</h3>
    </article>
    '''


def image_card(record: dict) -> str:
    reading = professional_reading(record)
    thumb_name = f"{Path(record['file_name']).stem}.jpg"
    chips = "".join([
        metric_chip("Family", record["family"]),
        metric_chip("Brightness", str(record["metrics"]["brightness_family"])),
        metric_chip("Texture", str(record["metrics"]["texture_family"])),
        metric_chip("Colorfulness", str(record["metrics"]["colorfulness"])),
        metric_chip("Dynamic range", str(record["metrics"]["dynamic_range"])),
        metric_chip("Sharpness", str(record["metrics"]["sharpness"])),
        metric_chip("Subjects", str(record["semantic"]["subject_count"])),
        metric_chip("Text", "yes" if record["semantic"]["text_present"] else "no"),
    ])
    objects = "".join(f"<li>{escape(item)}</li>" for item in record["semantic"].get("salient_objects", [])[:6])
    keywords = ", ".join(record["semantic"].get("keywords", [])[:8])
    title = record["semantic"].get("short_title") or record["file_name"]
    scene = record["semantic"].get("scene_summary") or "No semantic summary available."
    faithfulness = record["semantic"].get("faithfulness_score")
    overreach = record["semantic"].get("overreach_risk_score")
    qa = "n/a" if faithfulness is None else f"{faithfulness}/100"
    risk = "n/a" if overreach is None else f"{overreach}/100"
    return f'''
    <article class="image-card" id="{escape(Path(record["file_name"]).stem)}">
      <div class="image-visual">
        <img src="assets/thumbs/{escape(thumb_name)}" alt="{escape(record["file_name"])}">
      </div>
      <div class="image-copy">
        <p class="eyebrow">{escape(record["family"])}</p>
        <h3>{escape(title)}</h3>
        <p class="scene-summary">{escape(scene)}</p>
        <p class="curatorial">{escape(reading["curatorial_summary"])}</p>
        <div class="metrics-grid">{chips}</div>
        <div class="text-columns">
          <div>
            <h4>Professional reading</h4>
            <ul>
              {''.join(f'<li>{escape(item)}</li>' for item in reading["observations"][:6])}
            </ul>
          </div>
          <div>
            <h4>Semantic alignment</h4>
            <ul>
              <li>Faithfulness: {escape(qa)}</li>
              <li>Overreach risk: {escape(risk)}</li>
              <li>Support kind: {escape(str(record["semantic"].get("support_kind") or "n/a"))}</li>
              <li>Keywords: {escape(keywords or "n/a")}</li>
            </ul>
          </div>
        </div>
        <div class="objects-block">
          <h4>Salient objects</h4>
          <ul>{objects or '<li>n/a</li>'}</ul>
        </div>
      </div>
    </article>
    '''


def build_css() -> str:
    return """
:root {
  --bg: #f3ede3;
  --panel: #fbf7f1;
  --ink: #231f1b;
  --muted: #6f655b;
  --line: #d7cbbd;
  --accent: #9e5132;
  --accent-2: #36586b;
  --shadow: 0 18px 50px rgba(35, 31, 27, 0.08);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, rgba(158,81,50,0.12), transparent 28%),
    radial-gradient(circle at 85% 15%, rgba(54,88,107,0.10), transparent 24%),
    linear-gradient(180deg, #f6f1e8 0%, #efe8dc 100%);
}
a { color: inherit; }
.shell {
  width: min(1280px, calc(100vw - 48px));
  margin: 0 auto;
}
.hero {
  padding: 48px 0 32px;
}
.hero-grid {
  display: grid;
  grid-template-columns: 1.4fr 0.9fr;
  gap: 24px;
}
.hero-panel, .hero-aside, .section-panel, .image-card {
  background: rgba(251, 247, 241, 0.92);
  border: 1px solid var(--line);
  box-shadow: var(--shadow);
}
.hero-panel {
  padding: 34px;
  min-height: 420px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.hero-aside {
  padding: 22px;
  display: grid;
  gap: 18px;
  align-content: start;
}
.kicker, .eyebrow {
  font-family: "Courier New", monospace;
  font-size: 12px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
}
h1, h2, h3, h4 {
  font-weight: 500;
  margin: 0;
}
h1 {
  font-size: clamp(40px, 7vw, 88px);
  line-height: 0.95;
  letter-spacing: -0.04em;
  max-width: 11ch;
}
.lede {
  font-size: 18px;
  line-height: 1.7;
  color: var(--muted);
  max-width: 56ch;
}
.stat-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.stat-card {
  padding: 16px;
  border: 1px solid var(--line);
  background: rgba(255,255,255,0.55);
}
.stat-card strong {
  display: block;
  font-size: 28px;
  margin-top: 8px;
}
.nav-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 18px;
}
.nav-strip a {
  text-decoration: none;
  border: 1px solid var(--line);
  padding: 10px 14px;
  background: rgba(255,255,255,0.55);
}
.section {
  padding: 20px 0 28px;
}
.section-panel {
  padding: 26px;
}
.section-head {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: end;
  margin-bottom: 20px;
}
.section-head p {
  margin: 0;
  color: var(--muted);
  max-width: 58ch;
  line-height: 1.6;
}
.family-grid, .figure-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}
.family-card, .figure-card {
  border: 1px solid var(--line);
  background: rgba(255,255,255,0.55);
  padding: 18px;
}
.family-card strong {
  font-size: 24px;
  display: block;
  margin-top: 8px;
}
.figure-frame {
  background: #fff;
  border: 1px solid var(--line);
  aspect-ratio: 16 / 10;
  display: grid;
  place-items: center;
  overflow: hidden;
}
.figure-frame img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.trend-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.trend-block {
  border: 1px solid var(--line);
  background: rgba(255,255,255,0.55);
  padding: 18px;
}
.trend-block ul, .text-columns ul, .objects-block ul {
  margin: 12px 0 0;
  padding-left: 18px;
  line-height: 1.6;
}
.gallery {
  display: grid;
  gap: 22px;
}
.image-card {
  display: grid;
  grid-template-columns: 420px 1fr;
  overflow: hidden;
}
.image-visual {
  background: #dfd4c4;
  min-height: 420px;
}
.image-visual img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.image-copy {
  padding: 24px;
}
.scene-summary, .curatorial {
  line-height: 1.7;
  color: var(--muted);
}
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin: 18px 0;
}
.metric-chip {
  padding: 12px;
  border: 1px solid var(--line);
  background: rgba(255,255,255,0.62);
}
.metric-chip span {
  display: block;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--muted);
}
.metric-chip strong {
  display: block;
  margin-top: 6px;
  font-size: 17px;
}
.text-columns {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}
.objects-block {
  margin-top: 8px;
}
.footer {
  padding: 30px 0 60px;
  color: var(--muted);
  font-size: 14px;
}
@media (max-width: 1100px) {
  .hero-grid, .image-card, .trend-grid, .text-columns, .family-grid, .figure-grid, .metrics-grid {
    grid-template-columns: 1fr 1fr;
  }
  .image-card { grid-template-columns: 1fr; }
  .image-visual { min-height: 360px; }
}
@media (max-width: 760px) {
  .shell { width: min(100vw - 24px, 1000px); }
  .hero-grid, .trend-grid, .text-columns, .family-grid, .figure-grid, .metrics-grid { grid-template-columns: 1fr; }
  .hero-panel, .hero-aside, .section-panel, .image-copy { padding: 18px; }
  h1 { font-size: 48px; }
}
"""


def build_html(records: list[dict], summary: dict, semantic_trends: dict) -> str:
    families = summary.get("families", {})
    family_cards = "".join(
        f'''
        <article class="family-card">
          <p class="eyebrow">Family</p>
          <h3>{escape(label)}</h3>
          <strong>{count}</strong>
        </article>
        '''
        for label, count in list(families.items())[:9]
    )
    figures = "".join([
        figure_card("brightness_histogram.svg", "Brightness distribution"),
        figure_card("saturation_histogram.svg", "Saturation distribution"),
        figure_card("colorfulness_histogram.svg", "Colorfulness distribution"),
        figure_card("warm_balance_histogram.svg", "Warm balance distribution"),
        figure_card("sharpness_vs_dynamic_range.svg", "Sharpness vs dynamic range"),
        figure_card("saturation_vs_edge_density.svg", "Saturation vs edge density"),
        figure_card("visual_center_map.svg", "Visual center map"),
        figure_card("contact_sheet.png", "Corpus contact sheet"),
    ])
    objects = "".join(f"<li>{escape(label)} <strong>{count}</strong></li>" for label, count in semantic_trends.get("top_objects", [])[:10])
    keywords = "".join(f"<li>{escape(label)} <strong>{count}</strong></li>" for label, count in semantic_trends.get("top_keywords", [])[:10])
    gallery = "".join(image_card(record) for record in records)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Photo Corpus Study Lab</title>
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>
  <header class="hero">
    <div class="shell hero-grid">
      <section class="hero-panel">
        <div>
          <p class="kicker">Automated photographic study</p>
          <h1>Photo Corpus Study Lab</h1>
          <p class="lede">A static, automatically generated web report designed for photographers, editors, curators and research-oriented visual teams. It combines optical metrics, compositional signals and the semantic outputs of the main pipeline into one coherent reading surface.</p>
        </div>
        <nav class="nav-strip">
          <a href="#families">Families</a>
          <a href="#figures">Figures</a>
          <a href="#trends">Trends</a>
          <a href="#gallery">Image atlas</a>
        </nav>
      </section>
      <aside class="hero-aside">
        <div class="stat-grid">
          <div class="stat-card"><span class="eyebrow">Images</span><strong>{summary.get("image_count", 0)}</strong></div>
          <div class="stat-card"><span class="eyebrow">Human presence</span><strong>{summary.get("human_image_count", 0)}</strong></div>
          <div class="stat-card"><span class="eyebrow">Visible text</span><strong>{summary.get("text_image_count", 0)}</strong></div>
          <div class="stat-card"><span class="eyebrow">Mean colorfulness</span><strong>{summary.get("mean_colorfulness", "n/a")}</strong></div>
        </div>
        <div class="section-panel">
          <p class="eyebrow">Usage note</p>
          <p>This output is commercially usable as a decision-support and curation layer. For scientific-grade claims, it should be treated as a descriptive exploratory instrument unless separately benchmarked, calibrated and validated.</p>
        </div>
      </aside>
    </div>
  </header>

  <main>
    <section class="section" id="families">
      <div class="shell section-panel">
        <div class="section-head">
          <div>
            <p class="kicker">Corpus segmentation</p>
            <h2>Families</h2>
          </div>
          <p>The family labels align brightness, saturation, texture, human presence and text detection to create a practical first-pass clustering system across the corpus.</p>
        </div>
        <div class="family-grid">{family_cards}</div>
      </div>
    </section>

    <section class="section" id="figures">
      <div class="shell section-panel">
        <div class="section-head">
          <div>
            <p class="kicker">Visual diagnostics</p>
            <h2>Figures</h2>
          </div>
          <p>The figures below expose global tendencies of tone, color, structure and spatial organization across the image set.</p>
        </div>
        <div class="figure-grid">{figures}</div>
      </div>
    </section>

    <section class="section" id="trends">
      <div class="shell section-panel">
        <div class="section-head">
          <div>
            <p class="kicker">Cross-image alignment</p>
            <h2>Semantic and object trends</h2>
          </div>
          <p>These lists are derived from the main project outputs and aligned with the optical study. They are useful for iconographic grouping, edit logic and object recurrence tracking.</p>
        </div>
        <div class="trend-grid">
          <article class="trend-block">
            <p class="eyebrow">Recurring objects</p>
            <ul>{objects or '<li>n/a</li>'}</ul>
          </article>
          <article class="trend-block">
            <p class="eyebrow">Recurring keywords</p>
            <ul>{keywords or '<li>n/a</li>'}</ul>
          </article>
        </div>
      </div>
    </section>

    <section class="section" id="gallery">
      <div class="shell section-panel">
        <div class="section-head">
          <div>
            <p class="kicker">Per-image review</p>
            <h2>Image atlas</h2>
          </div>
          <p>Each card merges material reading, compositional metrics and semantic alignment to help professionals compare images without reopening raw intermediate files.</p>
        </div>
        <div class="gallery">{gallery}</div>
      </div>
    </section>
  </main>

  <footer class="footer">
    <div class="shell">
      Generated automatically by <code>photo_corpus_study_lab</code> from the image corpus and the main pipeline outputs.
    </div>
  </footer>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    site_dir = args.output_dir / "site"
    assets_dir = site_dir / "assets"
    thumbs_dir = assets_dir / "thumbs"
    figures_dir = assets_dir / "figures"
    ensure_dir(thumbs_dir)
    ensure_dir(figures_dir)

    records = []
    for path in sorted(args.metrics_dir.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    summary_payload = json.loads((args.output_dir / "families.json").read_text(encoding="utf-8"))
    summary = summary_payload.get("summary", {})
    semantic_trends = json.loads((args.output_dir / "semantic_trends.json").read_text(encoding="utf-8"))

    print(f"Building site for {len(records)} records", flush=True)
    copy_figures(args.output_dir / "figures", figures_dir)
    for index, record in enumerate(records, start=1):
        image_path = Path(record["image_path"])
        output_path = thumbs_dir / f"{image_path.stem}.jpg"
        build_thumbnail(image_path, output_path)
        print(f"[{index}/{len(records)}] {image_path.name}", flush=True)

    (assets_dir / "style.css").write_text(build_css(), encoding="utf-8")
    (site_dir / "index.html").write_text(build_html(records, summary, semantic_trends), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
