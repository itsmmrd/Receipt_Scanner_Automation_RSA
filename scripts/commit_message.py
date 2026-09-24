#!/usr/bin/env python3
"""Build a short commit title from the staged git diff."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def staged_files() -> list[str]:
    return [line for line in run("diff", "--cached", "--name-only").splitlines() if line]


def added_lines() -> list[str]:
    lines: list[str] = []
    for raw in run("diff", "--cached").splitlines():
        if raw.startswith("+") and not raw.startswith("+++"):
            text = raw[1:].strip()
            if text:
                lines.append(text)
    return lines


def looks_like_rename(added: list[str]) -> bool:
    return any("Receipt Scanner Automation RSA" in line for line in added)


def title_for_files(files: list[str], added: list[str]) -> str:
    names = {Path(path).name for path in files}

    if names == {"install.sh"}:
        joined = "\n".join(added)
        if "already_installed" in joined or "Uninstall" in joined or "/dev/tty" in joined:
            return "Fix installer token prompts and add reinstall/uninstall"
        if looks_like_rename(added):
            return "Rename installer to Receipt Scanner Automation RSA"
        if "raw.githubusercontent.com" in joined:
            return "Make the Ubuntu installer clone and set up from GitHub"
        return "Update Ubuntu installer"

    if names == {"README.md"}:
        joined = "\n".join(added)
        if "reinstall" in joined.lower() or "uninstall" in joined.lower():
            return "Document installer reinstall and uninstall options"
        if looks_like_rename(added):
            return "Rename README to Receipt Scanner Automation RSA"
        if "curl -fsSL" in joined:
            return "Add one-line Ubuntu install command to the README"
        return "Update project README"

    if names <= {"bot.py", "config.py"} and looks_like_rename(added):
        return "Rename project to Receipt Scanner Automation RSA"

    if names & {"install.sh", "README.md", ".gitignore"} and "curl -fsSL" in "\n".join(added):
        return "Add one-line Ubuntu install and keep tokens out of git"

    if names <= {"commit_message.py", "autocommit.sh", "auto-commit.sh"} or (
        "commit_message.py" in names
    ):
        return "Generate auto-commit messages from the actual changes"

    joined = "\n".join(added)
    if "bot.py" in names:
        if "normalize_edit_field" in joined or "EDIT_FIELD" in joined:
            return "Add AI-normalized button editing for receipt fields"
        if "manual_review_keyboard" in joined or "extract_receipt_from_text" in joined:
            return "Add AI manual text entry and photo attach flows"
        if "persist_review_session" in joined or "attach_photo_to_record" in joined:
            return "Keep photos during edits and link scans to existing rows"
        if "BOT_COMMANDS" in joined or "main_menu_keyboard" in joined:
            return "Add Telegram command menu and reply keyboard"
        if "edit_callback_message" in joined:
            return "Fix save-after-edit for text and photo review messages"
        return "Improve Telegram bot review and save flows"

    if "google_services.py" in names:
        if "code_verifier" in joined or "_pending_oauth" in joined:
            return "Fix Google OAuth PKCE session handling"
        if "attach_photo_to_record" in joined:
            return "Support attaching or replacing receipt photos in Drive"
        if "receipt_number" in joined:
            return "Use numeric receipt names and formatted sheet dates"
        return "Improve Google Drive and Sheets integration"

    if "extract_receipt.py" in names and "normalize_edit_field" in joined:
        return "Add Gemini helpers for text parse and field normalization"

    if names == {".env.example"} or (".env.example" in names and "README.md" in names):
        return "Document env vars and replace real server IP in README"

    if names == {".gitignore"} and ".env.local" in joined:
        return "Tighten gitignore for local env files"

    pretty = ", ".join(sorted(names))
    if len(files) == 1:
        return f"Update {pretty}"
    return f"Update {pretty}"


def _hunks(diff: str) -> list[tuple[str, str, list[str]]]:
    file_name = ""
    func = ""
    added: list[str] = []
    hunks: list[tuple[str, str, list[str]]] = []

    def flush() -> None:
        if file_name and added:
            hunks.append((file_name, func, list(added)))

    for line in diff.splitlines():
        if line.startswith("diff --git"):
            flush()
            file_name = ""
            func = ""
            added = []
        elif line.startswith("+++ b/"):
            file_name = line[6:]
        elif line.startswith("@@"):
            flush()
            added = []
            parts = line.split("@@")
            func = parts[2].strip() if len(parts) > 2 else ""
        elif line.startswith("+") and not line.startswith("+++"):
            text = line[1:].strip()
            if text and not text.startswith(("import ", "from ")):
                added.append(text)
    flush()
    return hunks


def _clean_line(line: str) -> str:
    text = line.strip().lstrip("-").strip().strip("`")
    if text.startswith(("```", "---", "|", "#")):
        return ""
    return text


def _bullet(file_name: str, func: str, added: list[str]) -> str:
    name = Path(file_name).name
    defs = [
        line.split("(", 1)[0].removeprefix("async ").strip()
        for line in added
        if line.startswith(("def ", "async def ", "class "))
    ]
    if defs:
        return f"{name}: {', '.join(defs[:3])}"
    for line in added:
        cleaned = _clean_line(line)
        if len(cleaned) >= 24:
            return f"{name}: {cleaned[:140]}"
    context = func if func.startswith(("def ", "class ", "async def ")) else ""
    where = f" in {context}" if context else ""
    snippet = _clean_line(added[0]) or added[0].strip()
    return f"{name}{where}: {snippet[:140]}"


def change_details() -> list[str]:
    diff = run("diff", "--cached", "--unified=3")
    bullets: list[str] = []
    seen: set[str] = set()
    for file_name, func, added in _hunks(diff):
        bullet = _bullet(file_name, func, added)
        if bullet in seen:
            continue
        seen.add(bullet)
        bullets.append(bullet)
        if len(bullets) == 8:
            break
    return bullets


def short_summary(files: list[str], added: list[str]) -> str:
    """One sentence about the change. Never a line of source code."""
    names = {Path(path).name for path in files}
    if names & {"commit_message.py", "auto-commit.mdc", "auto-commit.sh"}:
        return "Use a short sentence for each commit"
    title = title_for_files(files, added)
    if not title.startswith("Update "):
        return title
    if "receipt_pdf.py" in names or "Download PDF" in "\n".join(added):
        return "Save original photos and download them in a PDF"
    if names & {"scan_document.py", "document_segment.py"}:
        return "Straighten the receipt photo to the page frame"
    if names & {"google_services.py"}:
        return "Speed up Google Drive and Sheets calls"
    if names & {"bot.py"}:
        return "Adjust the Telegram receipt flow"
    if names & {"commit_message.py", "auto-commit.mdc", "auto-commit.sh"}:
        return "Use a short sentence for each commit"
    if names & {"requirements.txt"}:
        return "Update Python dependencies"
    if len(names) == 1:
        return f"Update {next(iter(names))}"
    return "Update " + ", ".join(sorted(names))


def main() -> None:
    files = staged_files()
    if not files:
        print("Update project files")
        return
    print(short_summary(files, added_lines()))


if __name__ == "__main__":
    main()
