"""Google OAuth, Drive uploads, and Sheets rows for RSA."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from config import (
    APP_NAME,
    DRIVE_FOLDER_NAME,
    OAUTH_DIR,
    SCOPES,
    SHEET_TITLE,
    require,
)

OAUTH_PENDING_TTL = 600
from receipt_format import parse_date_parts, receipt_number
from storage import load_user, save_user

SHEET_HEADERS = [
    "Name",
    "Date",
    "Category",
    "Amount",
    "Currency",
    "Merchant",
    "File link",
    "Created",
]

# Google Sheets custom display formats (real date values, not plain text).
SHEET_DATE_FORMAT = {"type": "DATE", "pattern": "dd.mm.yyyy"}
SHEET_CREATED_FORMAT = {"type": "DATE_TIME", "pattern": "dd.mm.yyyy hh:mm"}


def _sheet_date_formula(date_text: str | None) -> str:
    parts = parse_date_parts(date_text)
    if parts is None:
        return ""
    year, month, day = parts
    return f"=DATE({year},{month},{day})"


def _sheet_created_formula(when: datetime | None = None) -> str:
    now = when or datetime.now(timezone.utc)
    return (
        f"=DATE({now.year},{now.month},{now.day})"
        f"+TIME({now.hour},{now.minute},{now.second})"
    )


def _apply_sheet_date_formats(sheets, spreadsheet_id: str, sheet_id: int) -> None:
    """Format Date (B) and Created (H) columns as dates, not numbers/text."""
    requests = []
    for col_index, number_format in (
        (1, SHEET_DATE_FORMAT),
        (7, SHEET_CREATED_FORMAT),
    ):
        requests.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "startColumnIndex": col_index,
                        "endColumnIndex": col_index + 1,
                    },
                    "cell": {
                        "userEnteredFormat": {"numberFormat": number_format}
                    },
                    "fields": "userEnteredFormat.numberFormat",
                }
            }
        )
    sheets.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": requests},
    ).execute()


def public_base_url() -> str:
    return require("PUBLIC_BASE_URL").rstrip("/")


def redirect_uri() -> str:
    return f"{public_base_url()}/oauth/callback"


def connect_url(telegram_id: int) -> str:
    """Short link for Telegram buttons — redirects to Google OAuth."""
    return f"{public_base_url()}/oauth/start?uid={telegram_id}"


def normalize_callback_url(request_url: str) -> str:
    """Match Google token exchange to the registered redirect URI."""
    parsed = urlparse(request_url)
    base = redirect_uri()
    if parsed.query:
        return f"{base}?{parsed.query}"
    return base


def _pending_oauth_path(telegram_id: int) -> Path:
    return OAUTH_DIR / f"{telegram_id}.json"


def _save_pending_oauth(telegram_id: int, code_verifier: str) -> None:
    OAUTH_DIR.mkdir(parents=True, exist_ok=True)
    _pending_oauth_path(telegram_id).write_text(
        json.dumps({"code_verifier": code_verifier, "created": time.time()}),
        encoding="utf-8",
    )


def _load_pending_oauth(telegram_id: int) -> str:
    path = _pending_oauth_path(telegram_id)
    if not path.is_file():
        raise RuntimeError(
            "OAuth session expired or invalid. Send /connect in Telegram and open the new link."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    created = float(data.get("created") or 0)
    if time.time() - created > OAUTH_PENDING_TTL:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            "OAuth session expired. Send /connect in Telegram and open the new link."
        )
    verifier = data.get("code_verifier")
    if not verifier:
        raise RuntimeError(
            "OAuth session invalid. Send /connect in Telegram and open the new link."
        )
    return str(verifier)


def _clear_pending_oauth(telegram_id: int) -> None:
    _pending_oauth_path(telegram_id).unlink(missing_ok=True)


def _client_config() -> dict[str, Any]:
    return {
        "web": {
            "client_id": require("GOOGLE_CLIENT_ID"),
            "client_secret": require("GOOGLE_CLIENT_SECRET"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri()],
        }
    }


def authorization_url(telegram_id: int) -> str:
    flow = Flow.from_client_config(_client_config(), scopes=SCOPES)
    flow.redirect_uri = redirect_uri()
    url, _state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=str(telegram_id),
    )
    if flow.code_verifier:
        _save_pending_oauth(telegram_id, flow.code_verifier)
    return url


def credentials_from_record(record: dict[str, Any]) -> Credentials | None:
    data = record.get("google")
    if not data:
        return None
    creds = Credentials.from_authorized_user_info(data, scopes=SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        record["google"] = json_credentials(creds)
        save_user(record)
    return creds


def json_credentials(creds: Credentials) -> dict[str, Any]:
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or SCOPES),
    }


def finish_oauth(telegram_id: int, callback_url: str) -> dict[str, Any]:
    code_verifier = _load_pending_oauth(telegram_id)
    flow = Flow.from_client_config(_client_config(), scopes=SCOPES)
    flow.redirect_uri = redirect_uri()
    try:
        flow.fetch_token(
            authorization_response=normalize_callback_url(callback_url),
            code_verifier=code_verifier,
        )
    finally:
        _clear_pending_oauth(telegram_id)
    record = load_user(telegram_id)
    record["google"] = json_credentials(flow.credentials)
    save_user(record)
    ensure_google_workspace(telegram_id)
    return load_user(telegram_id)


_built_services: dict[str, tuple[str, Any]] = {}
_service_lock = threading.Lock()


def _google_service(name: str, version: str, creds: Credentials):
    """Reuse one API client per access token. Building a client downloads discovery docs."""
    token = creds.token or ""
    with _service_lock:
        cached = _built_services.get(name)
        if cached is not None and cached[0] == token:
            return cached[1]
        service = build(
            name,
            version,
            credentials=creds,
            cache_discovery=False,
            static_discovery=True,
        )
        _built_services[name] = (token, service)
        return service


def _services(record: dict[str, Any]):
    creds = credentials_from_record(record)
    if creds is None:
        raise RuntimeError("Google account is not connected.")
    return _google_service("drive", "v3", creds), _google_service("sheets", "v4", creds)


def ensure_google_workspace(telegram_id: int) -> dict[str, Any]:
    record = load_user(telegram_id)
    if record.get("folder_id") and record.get("spreadsheet_id") and record.get("sheet_ready"):
        return record
    drive, sheets = _services(record)
    if not record.get("folder_id"):
        created = (
            drive.files()
            .create(
                body={"name": DRIVE_FOLDER_NAME, "mimeType": "application/vnd.google-apps.folder"},
                fields="id",
            )
            .execute()
        )
        record["folder_id"] = created["id"]
    if not record.get("spreadsheet_id"):
        created = (
            sheets.spreadsheets()
            .create(
                body={"properties": {"title": SHEET_TITLE}},
                fields="spreadsheetId,spreadsheetUrl",
            )
            .execute()
        )
        record["spreadsheet_id"] = created["spreadsheetId"]
        record["spreadsheet_url"] = created.get("spreadsheetUrl")
        sheets.spreadsheets().values().update(
            spreadsheetId=record["spreadsheet_id"],
            range="A1:H1",
            valueInputOption="RAW",
            body={"values": [SHEET_HEADERS]},
        ).execute()
        drive.files().update(
            fileId=record["spreadsheet_id"],
            addParents=record["folder_id"],
            fields="id,parents",
        ).execute()
    meta = sheets.spreadsheets().get(spreadsheetId=record["spreadsheet_id"]).execute()
    sheet_id = meta["sheets"][0]["properties"]["sheetId"]
    header = (
        sheets.spreadsheets()
        .values()
        .get(spreadsheetId=record["spreadsheet_id"], range="A1:H1")
        .execute()
        .get("values")
        or [[]]
    )[0]
    if header != SHEET_HEADERS:
        sheets.spreadsheets().values().update(
            spreadsheetId=record["spreadsheet_id"],
            range="A1:H1",
            valueInputOption="RAW",
            body={"values": [SHEET_HEADERS]},
        ).execute()
    _apply_sheet_date_formats(sheets, record["spreadsheet_id"], sheet_id)
    record["sheet_ready"] = True
    save_user(record)
    return record


def next_receipt_name(telegram_id: int, date_text: str | None = None) -> str:
    record = load_user(telegram_id)
    number = int(record.get("next_number") or 1)
    return receipt_number(number)


def allocate_receipt_name(telegram_id: int, date_text: str | None = None) -> str:
    record = load_user(telegram_id)
    number = int(record.get("next_number") or 1)
    name = receipt_number(number)
    record["next_number"] = number + 1
    save_user(record)
    return name


def upload_receipt_file(telegram_id: int, path: Path, name: str) -> str:
    record = ensure_google_workspace(telegram_id)
    drive, _sheets = _services(record)
    media = MediaFileUpload(str(path), mimetype="image/jpeg", resumable=True)
    uploaded = (
        drive.files()
        .create(
            body={"name": f"{name}{path.suffix or '.jpg'}", "parents": [record["folder_id"]]},
            media_body=media,
            fields="id,webViewLink,webContentLink",
        )
        .execute()
    )
    drive.permissions().create(
        fileId=uploaded["id"],
        body={"type": "anyone", "role": "reader"},
    ).execute()
    return uploaded.get("webViewLink") or uploaded.get("webContentLink") or ""


def append_sheet_row(
    telegram_id: int,
    *,
    name: str,
    date: str | None,
    category: str | None,
    amount: float | None,
    currency: str | None,
    merchant: str | None,
    file_link: str,
) -> None:
    record = ensure_google_workspace(telegram_id)
    _drive, sheets = _services(record)
    sheets.spreadsheets().values().append(
        spreadsheetId=record["spreadsheet_id"],
        range="A:H",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body={
            "values": [
                [
                    name,
                    _sheet_date_formula(date),
                    category or "",
                    amount if amount is not None else "",
                    currency or "",
                    merchant or "",
                    file_link,
                    _sheet_created_formula(),
                ]
            ]
        },
    ).execute()


def list_records(telegram_id: int) -> list[list[str]]:
    record = load_user(telegram_id)
    if not record.get("spreadsheet_id"):
        record = ensure_google_workspace(telegram_id)
    creds = credentials_from_record(record)
    if creds is None:
        raise RuntimeError("Google account is not connected.")
    sheets = _google_service("sheets", "v4", creds)
    result = (
        sheets.spreadsheets()
        .values()
        .get(spreadsheetId=record["spreadsheet_id"], range="A2:H")
        .execute()
    )
    return result.get("values") or []


def _drive_image_files(drive, folder_id: str, name: str) -> list[dict[str, Any]]:
    query = (
        f"name contains '{name}' and '{folder_id}' in parents "
        "and trashed = false"
    )
    found = drive.files().list(q=query, fields="files(id,name,mimeType)").execute()
    images = []
    for item in found.get("files") or []:
        mime = item.get("mimeType") or ""
        if mime.startswith("image/") or mime == "application/pdf":
            images.append(item)
    return images


def attach_photo_to_record(telegram_id: int, name: str, path: Path) -> str:
    """Upload or replace the Drive photo for an existing sheet row."""
    record = ensure_google_workspace(telegram_id)
    drive, sheets = _services(record)
    rows = list_records(telegram_id)
    row_index = None
    for offset, row in enumerate(rows, start=2):
        if row and row[0] == name:
            row_index = offset
            break
    if row_index is None:
        raise ValueError(f"Record #{name} not found.")
    for item in _drive_image_files(drive, record["folder_id"], name):
        drive.files().delete(fileId=item["id"]).execute()
    file_link = upload_receipt_file(telegram_id, path, name)
    sheets.spreadsheets().values().update(
        spreadsheetId=record["spreadsheet_id"],
        range=f"G{row_index}",
        valueInputOption="RAW",
        body={"values": [[file_link]]},
    ).execute()
    return file_link


def delete_record(telegram_id: int, name: str) -> bool:
    record = ensure_google_workspace(telegram_id)
    drive, sheets = _services(record)
    rows = list_records(telegram_id)
    row_index = None
    file_link = ""
    for offset, row in enumerate(rows, start=2):
        if row and row[0] == name:
            row_index = offset
            file_link = row[6] if len(row) > 6 else ""
            break
    if row_index is None:
        return False
    meta = sheets.spreadsheets().get(spreadsheetId=record["spreadsheet_id"]).execute()
    sheet_id = meta["sheets"][0]["properties"]["sheetId"]
    sheets.spreadsheets().batchUpdate(
        spreadsheetId=record["spreadsheet_id"],
        body={
            "requests": [
                {
                    "deleteDimension": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "ROWS",
                            "startIndex": row_index - 1,
                            "endIndex": row_index,
                        }
                    }
                }
            ]
        },
    ).execute()
    for item in _drive_image_files(drive, record["folder_id"], name):
        drive.files().delete(fileId=item["id"]).execute()
    return True
