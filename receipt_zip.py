"""ZIP export: Excel sheet plus each receipt photo in a photos/ folder."""

from __future__ import annotations

import zipfile
from pathlib import Path

from config import SHEET_TITLE, TMP_DIR
from google_services import download_receipt_image, download_spreadsheet_xlsx, list_all_records


def _cell(row: list[str], index: int) -> str:
    if index >= len(row):
        return ""
    return str(row[index] or "")


def build_receipts_zip(telegram_id: int) -> Path:
    rows = list_all_records(telegram_id)
    if not rows:
        raise ValueError("There are no records yet.")
    folder = TMP_DIR / str(telegram_id) / "zip"
    folder.mkdir(parents=True, exist_ok=True)
    photos_dir = folder / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = folder / f"{SHEET_TITLE}.xlsx"
    download_spreadsheet_xlsx(telegram_id, xlsx_path)

    photo_paths: list[tuple[str, Path]] = []
    for row in rows:
        name = _cell(row, 0)
        if not name:
            continue
        dest = photos_dir / f"{name}.jpg"
        saved = download_receipt_image(telegram_id, name, dest)
        if saved is not None and saved.is_file():
            photo_paths.append((name, saved))

    zip_path = folder / "receipts.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(xlsx_path, arcname=xlsx_path.name)
        for name, path in photo_paths:
            suffix = path.suffix or ".jpg"
            archive.write(path, arcname=f"photos/{name}{suffix}")

    return zip_path
