"""PDF export: the receipt table first, then each saved photo labeled with its number."""

from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import TMP_DIR
from google_services import download_receipt_image, list_records

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 14 * mm
TABLE_HEADERS = ["#", "Date", "Category", "Amount", "Currency", "Merchant"]


def _cell(row: list[str], index: int) -> str:
    if index >= len(row):
        return ""
    return str(row[index] or "")


def _fit_image(path: Path, dest: Path) -> tuple[Path, float, float]:
    """Scale a photo so it fits one page without rewriting the Drive original."""
    max_w = PAGE_WIDTH - 2 * MARGIN
    max_h = PAGE_HEIGHT - 2 * MARGIN - 18 * mm
    with PILImage.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(max_w / width, max_h / height, 1.0)
        draw_w = width * scale
        draw_h = height * scale
        # Keep enough pixels for a printed page, and leave the Drive file untouched.
        pixel_scale = min(1600 / max(width, 1), 1600 / max(height, 1), 1.0)
        if pixel_scale < 1:
            image = image.resize(
                (max(1, int(width * pixel_scale)), max(1, int(height * pixel_scale))),
                PILImage.Resampling.LANCZOS,
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest, format="JPEG", quality=90)
    return dest, draw_w, draw_h


def render_receipts_pdf(
    rows: list[list[str]],
    images: dict[str, Path | None],
    dest: Path,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story: list = [
        Paragraph("Receipts", styles["Title"]),
        Spacer(1, 6 * mm),
    ]
    table_data = [TABLE_HEADERS]
    for row in rows:
        table_data.append(
            [
                _cell(row, 0),
                _cell(row, 1),
                _cell(row, 2),
                _cell(row, 3),
                _cell(row, 4),
                _cell(row, 5),
            ]
        )
    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2933")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d0d5dd")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(table)

    work = dest.parent / "pdf-images"
    for row in rows:
        name = _cell(row, 0)
        story.append(PageBreak())
        story.append(Paragraph(f"Receipt #{name}", styles["Heading1"]))
        story.append(Spacer(1, 4 * mm))
        image_path = images.get(name)
        if image_path is None or not image_path.is_file():
            story.append(Paragraph("No photo saved for this receipt.", styles["Normal"]))
            continue
        fitted, draw_w, draw_h = _fit_image(image_path, work / f"{name}.jpg")
        story.append(Image(str(fitted), width=draw_w, height=draw_h))

    doc = SimpleDocTemplate(
        str(dest),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="Receipts",
    )
    doc.build(story)
    return dest


def build_receipts_pdf(telegram_id: int) -> Path:
    rows = list_records(telegram_id)
    if not rows:
        raise ValueError("There are no records yet.")
    folder = TMP_DIR / str(telegram_id) / "pdf"
    folder.mkdir(parents=True, exist_ok=True)
    images: dict[str, Path | None] = {}
    for row in rows:
        name = _cell(row, 0)
        if not name:
            continue
        dest = folder / f"source-{name}"
        images[name] = download_receipt_image(telegram_id, name, dest)
    return render_receipts_pdf(rows, images, folder / "receipts.pdf")
