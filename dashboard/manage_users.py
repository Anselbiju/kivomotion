#!/usr/bin/env python3
"""Create and manage local KivoMotion coach and athlete accounts."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import secrets
import string
import tempfile
from pathlib import Path

from auth_server import DASHBOARD_DIR, DEFAULT_USERS_FILE, hash_password


def load_document(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "users": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_document(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def generated_password() -> str:
    alphabet = string.ascii_letters + string.digits + "-_.!@#"
    return "".join(secrets.choice(alphabet) for _ in range(20))


def read_password(args: argparse.Namespace) -> str:
    if args.generate_password:
        return generated_password()
    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Confirm password: "):
        raise SystemExit("Passwords do not match.")
    return password


def athlete_ids() -> set[str]:
    snapshot = json.loads((DASHBOARD_DIR / "src" / "data.json").read_text(encoding="utf-8"))
    return {row["athleteId"] for row in snapshot["queries"]["athletes"]["rows"]}


def upsert_user(path: Path, args: argparse.Namespace) -> None:
    email = args.email.strip().lower()
    password = read_password(args)
    role = "coach" if args.command == "add-coach" else "athlete"
    if role == "athlete" and args.athlete_id not in athlete_ids():
        raise SystemExit("The supplied athlete ID is not present in dashboard/src/data.json.")
    document = load_document(path)
    entry = {
        "email": email,
        "displayName": args.display_name.strip(),
        "role": role,
        "passwordHash": hash_password(password),
        **({"athleteId": args.athlete_id} if role == "athlete" else {}),
    }
    document["users"] = [user for user in document.get("users", []) if user.get("email", "").lower() != email]
    document["users"].append(entry)
    document["users"].sort(key=lambda user: user["email"])
    save_document(path, document)
    print(f"Saved {role} account: {email}")
    if args.generate_password:
        print(f"Generated password: {password}")


def remove_user(path: Path, email: str) -> None:
    document = load_document(path)
    normalized = email.strip().lower()
    before = len(document.get("users", []))
    document["users"] = [user for user in document.get("users", []) if user.get("email", "").lower() != normalized]
    if len(document["users"]) == before:
        raise SystemExit(f"No account found for {normalized}.")
    save_document(path, document)
    print(f"Removed account: {normalized}")


def list_users(path: Path) -> None:
    for user in load_document(path).get("users", []):
        suffix = f" · {user['athleteId']}" if user.get("athleteId") else ""
        print(f"{user['email']} · {user['role']} · {user.get('displayName', '')}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=Path, default=DEFAULT_USERS_FILE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("add-coach", "add-athlete"):
        child = subparsers.add_parser(command)
        child.add_argument("email")
        child.add_argument("--display-name", required=True)
        child.add_argument("--generate-password", action="store_true")
        if command == "add-athlete":
            child.add_argument("--athlete-id", required=True)
    remove = subparsers.add_parser("remove")
    remove.add_argument("email")
    subparsers.add_parser("list")
    args = parser.parse_args()
    if args.command in {"add-coach", "add-athlete"}:
        upsert_user(args.users, args)
    elif args.command == "remove":
        remove_user(args.users, args.email)
    else:
        list_users(args.users)


if __name__ == "__main__":
    main()
