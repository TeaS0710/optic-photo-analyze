#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, Preformatted, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a publishable static export bundle.")
    parser.add_argument("--main-output", type=Path, default=ROOT / "output")
    parser.add_argument("--study-output", type=Path, default=ROOT / "artifacts" / "photo_corpus_study")
    parser.add_argument("--export-dir", type=Path, default=ROOT / "artifacts" / "site_photo_analyze")
    parser.add_argument("--publish-dir", type=Path, default=ROOT / "public")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_files(patterns: list[tuple[Path, str]], destination: Path) -> list[Path]:
    copied: list[Path] = []
    ensure_dir(destination)
    for base, pattern in patterns:
        for path in sorted(base.glob(pattern)):
            if path.is_file():
                target = destination / path.name
                shutil.copy2(path, target)
                copied.append(target)
    return copied


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def markdown_inline_to_markup(text: str) -> str:
    text = escape_html(text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"__([^_]+)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"<i>\1</i>", text)
    return text


def build_pdf_from_markdown(md_path: Path, pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleExport",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        spaceAfter=10,
        textColor=colors.HexColor("#231f1b"),
    )
    heading_style = ParagraphStyle(
        "HeadingExport",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        spaceBefore=8,
        spaceAfter=6,
        textColor=colors.HexColor("#2e2a25"),
    )
    subheading_style = ParagraphStyle(
        "SubHeadingExport",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=15,
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.HexColor("#463e36"),
    )
    body_style = ParagraphStyle(
        "BodyExport",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        spaceAfter=6,
    )
    quote_style = ParagraphStyle(
        "QuoteExport",
        parent=body_style,
        leftIndent=14,
        rightIndent=8,
        textColor=colors.HexColor("#5a5046"),
        borderPadding=6,
        borderColor=colors.HexColor("#d6cdc2"),
        borderWidth=0.8,
        borderLeft=True,
        backColor=colors.HexColor("#faf6f0"),
    )
    code_style = ParagraphStyle(
        "CodeExport",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.5,
        leading=10.5,
        leftIndent=8,
        rightIndent=8,
        borderPadding=6,
        backColor=colors.HexColor("#f3eee7"),
    )

    story = []
    lines = md_path.read_text(encoding="utf-8").splitlines()
    in_code = False
    code_lines: list[str] = []
    bullet_items: list[ListItem] = []
    paragraph_lines: list[str] = []

    def flush_code() -> None:
        nonlocal code_lines
        if code_lines:
            story.append(Preformatted("\n".join(code_lines), code_style))
            story.append(Spacer(1, 4))
            code_lines = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            text = " ".join(part.strip() for part in paragraph_lines if part.strip())
            if text:
                story.append(Paragraph(markdown_inline_to_markup(text), body_style))
            paragraph_lines = []

    def flush_bullets() -> None:
        nonlocal bullet_items
        if bullet_items:
            story.append(ListFlowable(bullet_items, bulletType="bullet", start="circle", leftIndent=14))
            story.append(Spacer(1, 4))
            bullet_items = []

    for raw_line in lines:
        line = raw_line.rstrip()
        if line.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_paragraph()
                flush_bullets()
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not line.strip():
            flush_paragraph()
            flush_bullets()
            story.append(Spacer(1, 5))
            continue

        if line.startswith("# "):
            flush_paragraph()
            flush_bullets()
            story.append(Paragraph(markdown_inline_to_markup(line[2:].strip()), title_style))
            continue

        if line.startswith("## "):
            flush_paragraph()
            flush_bullets()
            story.append(Paragraph(markdown_inline_to_markup(line[3:].strip()), heading_style))
            continue

        if line.startswith("### "):
            flush_paragraph()
            flush_bullets()
            story.append(Paragraph(markdown_inline_to_markup(line[4:].strip()), subheading_style))
            continue

        if re.match(r"^\s*[-*]\s+", line):
            flush_paragraph()
            item_text = re.sub(r"^\s*[-*]\s+", "", line)
            bullet_items.append(ListItem(Paragraph(markdown_inline_to_markup(item_text.strip()), body_style)))
            continue

        if line.startswith("> "):
            flush_paragraph()
            flush_bullets()
            story.append(Paragraph(markdown_inline_to_markup(line[2:].strip()), quote_style))
            continue

        flush_bullets()
        paragraph_lines.append(line.replace("  ", "<br/>"))

    flush_paragraph()
    flush_bullets()
    flush_code()

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=md_path.stem,
    )
    doc.build(story)


def convert_markdowns_to_pdfs(root: Path) -> list[Path]:
    pdfs: list[Path] = []
    for md_path in sorted(root.rglob("*.md")):
        pdf_path = md_path.with_suffix(".pdf")
        build_pdf_from_markdown(md_path, pdf_path)
        pdfs.append(pdf_path)
    return pdfs


def append_site_styles(style_path: Path) -> None:
    extra_css = """
.nav-strip .nav-strong {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}
.library-hero {
  padding: 48px 0 22px;
}
.library-grid {
  display: grid;
  gap: 22px;
}
.library-block {
  background: rgba(251, 247, 241, 0.92);
  border: 1px solid var(--line);
  box-shadow: var(--shadow);
  padding: 26px;
}
.library-block.primary {
  background:
    linear-gradient(180deg, rgba(158,81,50,0.10), rgba(251,247,241,0.94)),
    rgba(251,247,241,0.94);
}
.library-block h2 {
  margin-bottom: 10px;
}
.library-block p {
  color: var(--muted);
  line-height: 1.7;
}
.pdf-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-top: 18px;
}
.pdf-card {
  border: 1px solid var(--line);
  background: rgba(255,255,255,0.62);
  padding: 16px;
}
.pdf-card strong {
  display: block;
  margin-bottom: 8px;
  font-size: 16px;
}
.pdf-card a {
  display: inline-block;
  margin-top: 10px;
  text-decoration: none;
  border-bottom: 1px solid currentColor;
}
@media (max-width: 760px) {
  .pdf-grid { grid-template-columns: 1fr; }
}
"""
    current = style_path.read_text(encoding="utf-8")
    if ".library-block" not in current:
        style_path.write_text(current + "\n" + extra_css, encoding="utf-8")


def build_pdf_library_page(export_dir: Path) -> None:
    site_dir = export_dir / "corpus_study_lab" / "site"
    assets_dir = site_dir / "assets"
    style_path = assets_dir / "style.css"
    append_site_styles(style_path)

    main_pdfs = sorted((export_dir / "main_pipeline").glob("*.pdf"))
    study_pdfs = sorted((export_dir / "corpus_study_lab").glob("*.pdf"))

    def cards(paths: list[Path], prefix: str) -> str:
        return "".join(
            f"""
            <article class="pdf-card">
              <p class="eyebrow">{escape_html(prefix)}</p>
              <strong>{escape_html(path.name)}</strong>
              <a href="{escape_html(os.path.relpath(path, site_dir))}">Open PDF</a>
            </article>
            """
            for path in paths
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PDF Library</title>
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>
  <header class="library-hero">
    <div class="shell">
      <section class="hero-panel">
        <div>
          <p class="kicker">Export library</p>
          <h1>PDF Access</h1>
          <p class="lede">This annex page centralizes the exported PDF reports. The main pipeline is foregrounded because it contains the primary analytical reports per image. The corpus study lab remains available as a secondary comparative layer.</p>
        </div>
        <nav class="nav-strip">
          <a href="index.html">Back to main site</a>
          <a class="nav-strong" href="library.html">PDF library</a>
        </nav>
      </section>
    </div>
  </header>
  <main class="section">
    <div class="shell library-grid">
      <section class="library-block primary">
        <p class="kicker">Primary category</p>
        <h2>Main Pipeline Reports</h2>
        <p>These PDFs correspond to the main analytical pipeline and should be treated as the primary report series for client-facing review, editorial reading and close per-image assessment.</p>
        <div class="pdf-grid">{cards(main_pdfs, "Main pipeline")}</div>
      </section>
      <section class="library-block">
        <p class="kicker">Secondary category</p>
        <h2>Corpus Study Lab Reports</h2>
        <p>These PDFs come from the corpus-level study layer. They are useful for trend analysis, comparative visual diagnostics and professional optical reading across the image set.</p>
        <div class="pdf-grid">{cards(study_pdfs, "Corpus study")}</div>
      </section>
    </div>
  </main>
  <footer class="footer">
    <div class="shell">
      Generated automatically by the unified root pipeline.
    </div>
  </footer>
</body>
</html>
"""
    (site_dir / "library.html").write_text(html, encoding="utf-8")

    index_path = site_dir / "index.html"
    index_html = index_path.read_text(encoding="utf-8")
    if 'href="library.html"' not in index_html:
        index_html = index_html.replace(
            '<a href="#gallery">Image atlas</a>',
            '<a href="#gallery">Image atlas</a>\n          <a class="nav-strong" href="library.html">PDF library</a>',
        )
        index_path.write_text(index_html, encoding="utf-8")


def build_manifest(export_dir: Path) -> None:
    files = []
    for path in sorted(export_dir.rglob("*")):
        if path.is_file():
            files.append({
                "path": str(path.relative_to(export_dir)),
                "size_bytes": path.stat().st_size,
            })
    (export_dir / "manifest.json").write_text(json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    export_dir = args.export_dir
    clean_dir(export_dir)

    copied_main = copy_files(
        [
            (args.main_output, "*.report.md"),
            (args.main_output, "*.analysis.json"),
            (args.main_output, "manifest.json"),
        ],
        export_dir / "main_pipeline",
    )
    copied_study_reports = copy_files(
        [
            (args.study_output, "corpus_report.md"),
            (args.study_output, "*.json"),
            (args.study_output / "reports", "*.md"),
            (args.study_output / "metrics", "*.json"),
        ],
        export_dir / "corpus_study_lab",
    )
    copy_tree(args.study_output / "site", export_dir / "corpus_study_lab" / "site")
    copy_tree(args.study_output / "figures", export_dir / "corpus_study_lab" / "figures")

    generated_pdfs = convert_markdowns_to_pdfs(export_dir)
    build_pdf_library_page(export_dir)
    build_manifest(export_dir)
    copy_tree(export_dir / "corpus_study_lab" / "site", args.publish_dir)

    summary = {
        "main_copied": len(copied_main),
        "study_copied": len(copied_study_reports),
        "pdf_generated": len(generated_pdfs),
        "publish_dir": str(args.publish_dir),
    }
    (export_dir / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
