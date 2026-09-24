"""Telegram bot for Receipt Scanner Automation RSA."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from aiohttp import web
from telegram import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
    MenuButtonCommands,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from config import APP_NAME, TMP_DIR, load_env_file, require
from extract_receipt import (
    ReceiptInfo,
    extract_receipt,
    extract_receipt_from_text,
    normalize_edit_field,
)
from google_services import (
    allocate_receipt_name,
    append_sheet_row,
    attach_photo_to_record,
    connect_url,
    delete_record,
    finish_oauth,
    list_records,
    next_receipt_name,
    public_base_url,
    redirect_uri,
    upload_receipt_file,
)
from receipt_format import (
    category_display,
    format_amount,
    format_display_date,
)
from receipt_pdf import build_receipts_pdf
from scan_document import ensure_telegram_photo_file
from storage import is_google_connected, load_user

load_env_file()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rsa")

REVIEW, EDIT_FIELD, ADD_TEXT, MANUAL_ATTACH, ATTACH_PICK, DELETE_NAME = range(6)

BOT_COMMANDS = [
    BotCommand("start", "Welcome and status"),
    BotCommand("connect", "Connect Google account"),
    BotCommand("add", "Paste receipt text — AI formats it"),
    BotCommand("text", "Same as /add — paste receipt text"),
    BotCommand("list", "Show recent receipts"),
    BotCommand("delete", "Remove a saved receipt"),
    BotCommand("cancel", "Cancel current action"),
    BotCommand("help", "How to use the bot"),
]

BTN_LIST = "📋 List"
BTN_ADD = "➕ Add"
BTN_REMOVE = "🗑 Remove"
BTN_SHEET = "📊 Sheet"
MENU_NAV_FILTER = filters.Regex(f"^({BTN_LIST}|{BTN_REMOVE}|{BTN_SHEET})$")


async def reply_processed_preview(
    message,
    processed: Path,
    *,
    caption: str,
    reply_markup: InlineKeyboardMarkup,
) -> None:
    ensure_telegram_photo_file(processed)
    try:
        with processed.open("rb") as handle:
            await message.reply_photo(
                photo=InputFile(handle, filename="receipt.jpg"),
                caption=caption,
                reply_markup=reply_markup,
            )
    except BadRequest as exc:
        if "photo" not in str(exc).lower():
            raise
        log.warning("reply_photo rejected (%s); sending as document", exc)
        with processed.open("rb") as handle:
            await message.reply_document(
                document=InputFile(handle, filename="receipt.jpg"),
                caption=caption,
                reply_markup=reply_markup,
            )


async def send_google_connect(update: Update) -> None:
    target = update.effective_message
    if target is None or update.effective_user is None:
        return
    try:
        url = connect_url(update.effective_user.id)
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Connect Google account", url=url)]]
        )
        await target.reply_text(
            f"Welcome to {APP_NAME}.\n\n"
            "Connect your Google account to save receipts to Drive and Sheets.",
            reply_markup=keyboard,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Could not build Google connect link")
        await target.reply_text(
            f"{APP_NAME} is online, but Google connect failed:\n{exc}\n\n"
            "Check GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and PUBLIC_BASE_URL "
            "in /opt/rsa/.env"
        )


def require_google(handler):
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if user is None:
            return ConversationHandler.END
        if not is_google_connected(user.id):
            await send_google_connect(update)
            return ConversationHandler.END
        return await handler(update, context)

    return wrapped


def main_menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [[BTN_LIST, BTN_ADD], [BTN_REMOVE, BTN_SHEET]],
        resize_keyboard=True,
    )


def links_keyboard(file_link: str | None, sheet_url: str | None) -> InlineKeyboardMarkup | None:
    buttons = []
    if file_link:
        buttons.append(InlineKeyboardButton("Open Drive", url=file_link))
    if sheet_url:
        buttons.append(InlineKeyboardButton("Open Sheet", url=sheet_url))
    if not buttons:
        return None
    return InlineKeyboardMarkup([buttons])


def records_keyboard(
    rows: list[list[str]],
    sheet_url: str | None,
    *,
    delete_only: bool = False,
) -> InlineKeyboardMarkup:
    buttons: list[list[InlineKeyboardButton]] = []
    for row in rows[-10:]:
        name = row[0]
        date = row[1] if len(row) > 1 else ""
        amount = row[3] if len(row) > 3 else ""
        label = f"#{name} · {date} · {amount}".strip()
        file_link = row[6] if len(row) > 6 else ""
        if delete_only:
            buttons.append(
                [InlineKeyboardButton(f"🗑 {label}", callback_data=f"del:{name}")]
            )
        else:
            row_buttons = []
            if file_link:
                row_buttons.append(InlineKeyboardButton(f"📁 #{name}", url=file_link))
            row_buttons.append(
                InlineKeyboardButton(f"🗑 #{name}", callback_data=f"del:{name}")
            )
            buttons.append(row_buttons)
    if sheet_url:
        buttons.append([InlineKeyboardButton("Open Sheet", url=sheet_url)])
    if not delete_only and rows:
        buttons.append([InlineKeyboardButton("Download PDF", callback_data="download_pdf")])
    return InlineKeyboardMarkup(buttons)


def format_info(info: ReceiptInfo, name: str | None = None) -> str:
    lines = []
    if name:
        lines.append(f"#{name}")
    lines.extend(
        [
            f"Date: {format_display_date(info.date)}",
            f"Category: {category_display(info)}",
            f"Amount: {format_amount(info)}",
            f"Merchant: {info.merchant or 'unknown'}",
        ]
    )
    return "\n".join(lines)


async def edit_callback_message(
    query: CallbackQuery,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    """Update the review message whether it is a photo caption or plain text."""
    if query.message and query.message.photo:
        await query.edit_message_caption(caption=text, reply_markup=reply_markup)
    else:
        await query.edit_message_text(text=text, reply_markup=reply_markup)


def format_saved_message(info: ReceiptInfo, name: str) -> str:
    return (
        f"Saved #{name}\n\n"
        f"Date: {format_display_date(info.date)}\n"
        f"Category: {category_display(info)}\n"
        f"Amount: {format_amount(info)}\n"
        f"Merchant: {info.merchant or 'unknown'}"
    )


def edit_text_button() -> InlineKeyboardButton:
    return InlineKeyboardButton("Edit text", callback_data="edit_text")


def edit_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Date", callback_data="edit:date"),
                InlineKeyboardButton("Category", callback_data="edit:category"),
            ],
            [
                InlineKeyboardButton("Amount", callback_data="edit:amount"),
                InlineKeyboardButton("Currency", callback_data="edit:currency"),
            ],
            [
                InlineKeyboardButton("Merchant", callback_data="edit:merchant"),
                InlineKeyboardButton("Time", callback_data="edit:time"),
            ],
            [InlineKeyboardButton("Done", callback_data="edit:back")],
        ]
    )


def review_keyboard(info: ReceiptInfo) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Save", callback_data="save_original")],
            [InlineKeyboardButton("Link to existing", callback_data="link_existing")],
            [
                edit_text_button(),
                InlineKeyboardButton("Cancel", callback_data="cancel"),
            ],
        ]
    )


def manual_review_keyboard(info: ReceiptInfo) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Save without photo", callback_data="save_text_only")],
            [InlineKeyboardButton("Attach photo", callback_data="attach_photo")],
            [
                edit_text_button(),
                InlineKeyboardButton("Cancel", callback_data="cancel"),
            ],
        ]
    )


def active_review_keyboard(context: ContextTypes.DEFAULT_TYPE) -> InlineKeyboardMarkup:
    info = info_from_context(context)
    if context.user_data.get("manual") and not context.user_data.get("has_photo"):
        return manual_review_keyboard(info)
    return review_keyboard(info)


def attach_pick_keyboard(rows: list[list[str]]) -> InlineKeyboardMarkup:
    buttons: list[list[InlineKeyboardButton]] = []
    for row in rows[-15:]:
        name = row[0]
        has_photo = bool(row[6] if len(row) > 6 else "")
        tag = "replace photo" if has_photo else "add photo"
        date = row[1] if len(row) > 1 else ""
        amount = row[3] if len(row) > 3 else ""
        label = f"#{name} · {date} · {amount} · {tag}"
        buttons.append(
            [InlineKeyboardButton(label, callback_data=f"attach:{name}")]
        )
    buttons.append([InlineKeyboardButton("Cancel", callback_data="attach:cancel")])
    return InlineKeyboardMarkup(buttons)


MEDIA_KEYS = ("original", "processed", "manual", "link_use_original", "has_photo")


def backup_media(context: ContextTypes.DEFAULT_TYPE) -> dict[str, object]:
    return {key: context.user_data[key] for key in MEDIA_KEYS if key in context.user_data}


def restore_media(context: ContextTypes.DEFAULT_TYPE, saved: dict[str, object]) -> None:
    context.user_data.update(saved)


SESSIONS_KEY = "rsa_review_sessions"


def persist_review_session(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> None:
    sessions = context.bot_data.setdefault(SESSIONS_KEY, {})
    session: dict[str, object] = {}
    if "info" in context.user_data:
        session["info"] = context.user_data["info"]
    for key in MEDIA_KEYS:
        if key in context.user_data:
            session[key] = context.user_data[key]
    sessions[user_id] = session


def restore_review_session(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> None:
    sessions = context.bot_data.get(SESSIONS_KEY, {})
    session = sessions.get(user_id, {})
    if session.get("info"):
        context.user_data["info"] = session["info"]
    for key in MEDIA_KEYS:
        if key in session:
            context.user_data[key] = session[key]


def info_from_context(context: ContextTypes.DEFAULT_TYPE) -> ReceiptInfo:
    data = context.user_data.get("info") or {}
    return ReceiptInfo.model_validate(data)


def store_info(context: ContextTypes.DEFAULT_TYPE, info: ReceiptInfo) -> None:
    media = backup_media(context)
    context.user_data["info"] = info.model_dump()
    restore_media(context, media)


def resolve_photo_path(
    context: ContextTypes.DEFAULT_TYPE, *, prefer_original: bool
) -> Path | None:
    primary = "original" if prefer_original else "processed"
    fallback = "processed" if prefer_original else "original"
    for key in (primary, fallback):
        path_str = context.user_data.get(key)
        if path_str and Path(path_str).is_file():
            return Path(path_str)
    return None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.effective_user is None:
        return
    if not is_google_connected(update.effective_user.id):
        await send_google_connect(update)
        return
    await update.message.reply_text(
        f"{APP_NAME} is ready.\n\n"
        "Send a receipt photo to scan it, or use the menu below.",
        reply_markup=main_menu_keyboard(),
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        lines = [f"{APP_NAME}", "", "Commands:"]
        lines.extend(f"/{cmd.command} — {cmd.description}" for cmd in BOT_COMMANDS)
        lines.extend(["", "Or send a receipt photo to scan it."])
        await update.message.reply_text("\n".join(lines))


async def connect_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.effective_user is None:
        return
    url = connect_url(update.effective_user.id)
    await update.message.reply_text(
        "Connect or refresh your Google account:",
        reply_markup=InlineKeyboardMarkup(
            [[InlineKeyboardButton("Connect Google account", url=url)]]
        ),
    )


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


async def review_saved_upload(
    message,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    original: Path,
) -> int:
    """Read the uploaded file as-is. The saved copy is this file, not a crop."""
    await message.reply_text("Reading receipt...")
    try:
        info = await asyncio.to_thread(extract_receipt, original)
    except Exception as exc:  # noqa: BLE001
        log.exception("Extract failed")
        await message.reply_text(f"Could not process this photo: {exc}")
        return ConversationHandler.END

    preview = original.with_name(f"preview{original.suffix}")
    shutil.copyfile(original, preview)
    store_info(context, info)
    context.user_data["original"] = str(original)
    context.user_data["processed"] = str(original)
    context.user_data["has_photo"] = True
    persist_review_session(context, user_id)
    name = next_receipt_name(user_id, info.date)
    await reply_processed_preview(
        message,
        preview,
        caption=f"{format_info(info, name)}\n\nSave this photo?",
        reply_markup=review_keyboard(info),
    )
    return REVIEW


@require_google
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    message = update.message
    photo = message.photo[-1]
    user_dir = TMP_DIR / str(update.effective_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    original = user_dir / "original.jpg"
    file = await photo.get_file()
    await file.download_to_drive(original)
    return await review_saved_upload(message, context, update.effective_user.id, original)


@require_google
async def handle_image_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    message = update.message
    document = message.document
    suffix = Path(document.file_name or "receipt.jpg").suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        await message.reply_text("Send a photo or an image file.")
        return ConversationHandler.END
    user_dir = TMP_DIR / str(update.effective_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    original = user_dir / f"original{suffix}"
    file = await document.get_file()
    await file.download_to_drive(original)
    return await review_saved_upload(message, context, update.effective_user.id, original)


async def review_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    action = query.data
    if action == "cancel":
        await edit_callback_message(query, "Cancelled. Nothing was saved.")
        return ConversationHandler.END
    if action == "edit_text":
        persist_review_session(context, query.from_user.id)
        await query.message.reply_text(
            "Tap a field to edit:",
            reply_markup=edit_keyboard(),
        )
        return EDIT_FIELD
    if action == "link_existing":
        rows = await asyncio.to_thread(list_records, update.effective_user.id)
        if not rows:
            await query.message.reply_text("No saved receipts yet.")
            return REVIEW
        await query.message.reply_text(
            "Link this photo to which receipt?",
            reply_markup=attach_pick_keyboard(rows),
        )
        return ATTACH_PICK
    if action == "attach_photo":
        await query.message.reply_text("Send the receipt photo to attach.")
        return MANUAL_ATTACH
    if action in {"save_result", "save_original"}:
        await save_current(update, context, use_original=True, photo_required=True)
        return ConversationHandler.END
    return REVIEW


EDIT_FIELD_HINTS = {
    "date": "Send a date in any format (e.g. 1.9.26). AI will fix it. Send - to clear.",
    "category": "Send a category (any spelling). AI will clean it. Send - to clear.",
    "amount": "Send an amount (e.g. 31,31 euro). AI will fix it. Send - to clear.",
    "currency": "Send a currency (e.g. euro). AI will fix it. Send - to clear.",
    "merchant": "Send the store name. AI will fix spelling. Send - to clear.",
    "time": "Send a time (e.g. 7pm). AI will fix it. Send - to clear.",
}

EDIT_FIELD_ATTR = {
    "date": "date",
    "category": "category",
    "amount": "amount",
    "currency": "currency",
    "merchant": "merchant",
    "time": "purchase_time",
}


async def edit_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    field = query.data.split(":", 1)[1]
    if field == "back":
        restore_review_session(context, query.from_user.id)
        info = info_from_context(context)
        name = next_receipt_name(query.from_user.id, info.date)
        await query.message.reply_text(
            f"{format_info(info, name)}\n\nIs this result OK?",
            reply_markup=active_review_keyboard(context),
        )
        return REVIEW
    context.user_data["edit_field"] = field
    await query.message.reply_text(EDIT_FIELD_HINTS.get(field, f"Send the new {field}."))
    return EDIT_FIELD


async def edit_value(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message is None or update.effective_user is None:
        return EDIT_FIELD
    field = context.user_data.get("edit_field")
    if not field:
        await update.message.reply_text("Tap a field to edit:", reply_markup=edit_keyboard())
        return EDIT_FIELD
    info = info_from_context(context)
    text = update.message.text.strip()
    if text == "-":
        setattr(info, EDIT_FIELD_ATTR[field], None)
    else:
        await update.message.reply_text("Formatting with AI...")
        try:
            normalized = await asyncio.to_thread(normalize_edit_field, field, text)
        except Exception as exc:  # noqa: BLE001
            log.exception("Field normalize failed")
            await update.message.reply_text(f"Could not understand that: {exc}\n\nTry again.")
            return EDIT_FIELD
        setattr(info, EDIT_FIELD_ATTR[field], normalized)
    store_info(context, info)
    persist_review_session(context, update.effective_user.id)
    context.user_data.pop("edit_field", None)
    name = next_receipt_name(update.effective_user.id, info.date)
    await update.message.reply_text(
        f"Updated {field}.\n\n{format_info(info, name)}\n\n"
        "Tap another field or Done.",
        reply_markup=edit_keyboard(),
    )
    return EDIT_FIELD


async def save_current(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    use_original: bool,
    photo_required: bool,
) -> None:
    user_id = update.effective_user.id
    info = info_from_context(context)
    name = allocate_receipt_name(user_id, info.date)
    file_link = ""
    if photo_required:
        image_path = resolve_photo_path(context, prefer_original=use_original)
        if image_path is None:
            message = update.effective_message
            text = "Photo not found. Scan the receipt again."
            if update.callback_query:
                await update.callback_query.message.reply_text(text)
            elif message:
                await message.reply_text(text)
            return
        file_link = await asyncio.to_thread(upload_receipt_file, user_id, image_path, name)
    await asyncio.to_thread(
        append_sheet_row,
        user_id,
        name=name,
        date=info.date,
        category=category_display(info),
        amount=info.amount,
        currency=info.currency,
        merchant=info.merchant,
        file_link=file_link,
    )
    record = load_user(user_id)
    text = format_saved_message(info, name)
    keyboard = links_keyboard(file_link, record.get("spreadsheet_url"))
    message = update.effective_message
    if update.callback_query:
        await edit_callback_message(update.callback_query, text, reply_markup=keyboard)
    elif message:
        await message.reply_text(text, reply_markup=keyboard)


@require_google
async def add_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    context.user_data["manual"] = True
    await update.message.reply_text(
        "Paste everything in one message — date, store, amount, category, notes.\n"
        "AI will clean and format it."
    )
    return ADD_TEXT


@require_google
async def text_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await add_start(update, context)


async def add_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    await update.message.reply_text("Formatting with AI...")
    try:
        info = await asyncio.to_thread(extract_receipt_from_text, text)
    except Exception as exc:  # noqa: BLE001
        log.exception("Text parse failed")
        await update.message.reply_text(f"Could not parse that text: {exc}\n\nTry again.")
        return ADD_TEXT
    store_info(context, info)
    persist_review_session(context, update.effective_user.id)
    name = next_receipt_name(update.effective_user.id, info.date)
    await update.message.reply_text(
        f"Parsed receipt:\n\n{format_info(info, name)}\n\nIs this correct?",
        reply_markup=manual_review_keyboard(info),
    )
    return REVIEW


async def manual_attach_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_dir = TMP_DIR / str(update.effective_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    original = user_dir / "original.jpg"
    file = await update.message.photo[-1].get_file()
    await file.download_to_drive(original)
    context.user_data["original"] = str(original)
    context.user_data["processed"] = str(original)
    context.user_data["has_photo"] = True
    await save_current(update, context, use_original=True, photo_required=True)
    return ConversationHandler.END


async def attach_pick_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "attach:cancel":
        name = next_receipt_name(update.effective_user.id, info_from_context(context).date)
        await query.message.reply_text(
            f"{format_info(info_from_context(context), name)}\n\nIs this result OK?",
            reply_markup=active_review_keyboard(context),
        )
        return REVIEW
    name = query.data.split(":", 1)[1]
    user_id = update.effective_user.id
    path = resolve_photo_path(context, prefer_original=bool(context.user_data.get("link_use_original")))
    if path is None:
        await query.message.reply_text("Photo not found. Scan again.")
        return ConversationHandler.END
    try:
        file_link = await asyncio.to_thread(attach_photo_to_record, user_id, name, path)
    except Exception as exc:  # noqa: BLE001
        log.exception("Attach photo failed")
        await query.message.reply_text(f"Could not attach photo: {exc}")
        return ConversationHandler.END
    record = load_user(user_id)
    await query.message.reply_text(
        f"Photo linked to #{name}.",
        reply_markup=links_keyboard(file_link, record.get("spreadsheet_url")),
    )
    return ConversationHandler.END


async def save_text_only_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await save_current(update, context, use_original=False, photo_required=False)
    return ConversationHandler.END


@require_google
async def delete_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await show_remove_menu(update, context)
    return ConversationHandler.END


async def delete_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    name = update.message.text.strip().lstrip("#")
    removed = await asyncio.to_thread(delete_record, update.effective_user.id, name)
    if removed:
        await update.message.reply_text(f"Removed #{name} from the sheet and Drive.")
    else:
        await update.message.reply_text(f"No record #{name} was found.")
    return ConversationHandler.END


async def show_records_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    target = update.effective_message
    if target is None or update.effective_user is None:
        return
    rows = await asyncio.to_thread(list_records, update.effective_user.id)
    if not rows:
        await target.reply_text("There are no records yet.", reply_markup=main_menu_keyboard())
        return
    record = load_user(update.effective_user.id)
    sheet_url = record.get("spreadsheet_url")
    preview = "\n".join(
        f"#{row[0]} · {row[1] if len(row) > 1 else ''} · "
        f"{row[2] if len(row) > 2 else ''} · {row[3] if len(row) > 3 else ''}"
        for row in rows[-10:]
    )
    await target.reply_text(
        f"Recent receipts:\n{preview}",
        reply_markup=records_keyboard(rows, sheet_url),
    )


async def show_remove_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    target = update.effective_message
    if target is None or update.effective_user is None:
        return
    rows = await asyncio.to_thread(list_records, update.effective_user.id)
    if not rows:
        await target.reply_text("There are no records yet.", reply_markup=main_menu_keyboard())
        return
    record = load_user(update.effective_user.id)
    await target.reply_text(
        "Tap a receipt to remove it from the sheet and Drive:",
        reply_markup=records_keyboard(rows, record.get("spreadsheet_url"), delete_only=True),
    )


@require_google
async def list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await show_records_list(update, context)


@require_google
async def sheet_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    target = update.effective_message
    if target is None or update.effective_user is None:
        return
    record = load_user(update.effective_user.id)
    sheet_url = record.get("spreadsheet_url")
    if not sheet_url:
        await target.reply_text("No sheet yet. Save a receipt first.")
        return
    await target.reply_text(
        "Your RSA spreadsheet:",
        reply_markup=InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("Open Sheet", url=sheet_url)],
                [InlineKeyboardButton("Download PDF", callback_data="download_pdf")],
            ]
        ),
    )


@require_google
async def download_pdf_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await query.message.reply_text("Building the PDF...")
    try:
        path = await asyncio.to_thread(build_receipts_pdf, query.from_user.id)
    except Exception as exc:  # noqa: BLE001
        log.exception("PDF export failed")
        await query.message.reply_text(f"Could not build the PDF: {exc}")
        return
    with path.open("rb") as handle:
        await query.message.reply_document(
            document=InputFile(handle, filename="receipts.pdf"),
            caption="The table is first. Each photo follows, labeled with its receipt number.",
        )


@require_google
async def menu_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    text = update.message.text.strip()
    if text == BTN_LIST:
        await show_records_list(update, context)
    elif text == BTN_REMOVE:
        await show_remove_menu(update, context)
    elif text == BTN_SHEET:
        await sheet_cmd(update, context)


async def delete_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or update.effective_user is None:
        return
    await query.answer()
    if not is_google_connected(update.effective_user.id):
        await query.message.reply_text("Connect Google first with /connect.")
        return
    name = query.data.split(":", 1)[1]
    removed = await asyncio.to_thread(delete_record, update.effective_user.id, name)
    if removed:
        await query.message.reply_text(f"Removed #{name} from the sheet and Drive.")
    else:
        await query.message.reply_text(f"No record #{name} was found.")


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END


async def oauth_health(_request: web.Request) -> web.Response:
    return web.Response(text=f"{APP_NAME} OAuth server is running.\n", content_type="text/plain")


async def oauth_start(request: web.Request) -> web.Response:
    from google_services import authorization_url

    uid = request.query.get("uid", "").strip()
    if not uid.isdigit():
        return web.Response(text="Missing or invalid uid", status=400)
    raise web.HTTPFound(authorization_url(int(uid)))


async def oauth_callback(request: web.Request) -> web.Response:
    error = request.query.get("error")
    if error:
        detail = request.query.get("error_description", error)
        return web.Response(
            text=f"Google OAuth failed: {detail}\n\nCheck redirect URI:\n{redirect_uri()}",
            status=400,
            content_type="text/plain",
        )
    state = request.query.get("state", "").strip()
    if not state.isdigit():
        return web.Response(text="Invalid OAuth state", status=400)
    try:
        finish_oauth(int(state), str(request.url))
    except Exception as exc:  # noqa: BLE001
        log.exception("OAuth callback failed")
        return web.Response(
            text=(
                f"OAuth failed: {exc}\n\n"
                f"Make sure this exact redirect URI is in Google Cloud Console:\n"
                f"{redirect_uri()}"
            ),
            status=500,
            content_type="text/plain",
        )
    return web.Response(
        text=(
            f"{APP_NAME} is connected. You can close this tab and return to Telegram. "
            "Send /start to the bot."
        ),
        content_type="text/plain",
    )


async def setup_bot_menu(app: Application) -> None:
    await app.bot.set_my_commands(BOT_COMMANDS)
    await app.bot.set_chat_menu_button(menu_button=MenuButtonCommands())
    log.info("Bot command menu registered (%d commands)", len(BOT_COMMANDS))


async def on_startup(app: Application) -> None:
    me = await app.bot.get_me()
    log.info("Telegram connected as @%s (id=%s)", me.username, me.id)
    await setup_bot_menu(app)
    try:
        await start_oauth_site(app)
    except Exception:
        log.exception("OAuth HTTP server failed; Telegram polling still runs")


def oauth_listen_port() -> int:
    """Internal port for OAuth HTTP. Public HTTPS is handled by Caddy."""
    return int(os.environ.get("OAUTH_PORT", "8090"))


async def start_oauth_site(app: Application) -> None:
    host = "0.0.0.0"
    listen_port = oauth_listen_port()
    site = web.Application()
    site.router.add_get("/oauth/health", oauth_health)
    site.router.add_get("/oauth/start", oauth_start)
    site.router.add_get("/oauth/callback", oauth_callback)
    runner = web.AppRunner(site)
    await runner.setup()
    listener = web.TCPSite(runner, host=host, port=listen_port)
    await listener.start()
    log.info("OAuth server listening on %s:%s", host, listen_port)
    app.bot_data["oauth_runner"] = runner


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("Telegram handler failed: %s", context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            f"Something went wrong: {context.error}"
        )


def build_application() -> Application:
    token = require("TELEGRAM_BOT_TOKEN")
    application = Application.builder().token(token).post_init(on_startup).build()
    application.add_error_handler(on_error)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("connect", connect_cmd))
    application.add_handler(CommandHandler("list", list_cmd))
    review_conv = ConversationHandler(
        entry_points=[
            MessageHandler(filters.PHOTO, handle_photo),
            MessageHandler(filters.Document.IMAGE, handle_image_file),
            CommandHandler("add", add_start),
            CommandHandler("text", text_start),
            CommandHandler("delete", delete_start),
            MessageHandler(filters.Regex(f"^{BTN_ADD}$"), add_start),
        ],
        states={
            REVIEW: [
                CallbackQueryHandler(save_text_only_callback, pattern="^save_text_only$"),
                CallbackQueryHandler(review_callback),
            ],
            EDIT_FIELD: [
                CallbackQueryHandler(edit_choice, pattern=r"^edit:"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, edit_value),
            ],
            ADD_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_text_input)],
            MANUAL_ATTACH: [MessageHandler(filters.PHOTO, manual_attach_photo)],
            ATTACH_PICK: [CallbackQueryHandler(attach_pick_callback, pattern=r"^attach:")],
            DELETE_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, delete_name)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        allow_reentry=True,
    )

    application.add_handler(review_conv)
    application.add_handler(MessageHandler(MENU_NAV_FILTER, menu_router))
    application.add_handler(CallbackQueryHandler(delete_callback, pattern=r"^del:"))
    application.add_handler(CallbackQueryHandler(download_pdf_callback, pattern=r"^download_pdf$"))
    return application


def main() -> None:
    try:
        application = build_application()
        log.info("Starting %s", APP_NAME)
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
    except Exception:
        log.exception("%s failed to start", APP_NAME)
        raise


if __name__ == "__main__":
    main()
