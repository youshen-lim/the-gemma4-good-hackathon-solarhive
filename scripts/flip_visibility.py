#!/usr/bin/env python3
"""Flip SolarHive HF repo visibility for hackathon submission.

Atomically toggles all SolarHive HF repos (1 dataset + 5 models + 1 space)
between private and public. Defaults to dry-run; pass --execute to actually
update visibility.

Usage:
    # Preview what would change (default — no API mutations):
    HF_TOKEN=xxx python scripts/flip_visibility.py --make public

    # Execute the flip:
    HF_TOKEN=xxx python scripts/flip_visibility.py --make public --execute

    # Just inspect current state without changing anything:
    HF_TOKEN=xxx python scripts/flip_visibility.py --verify-only

Note: GitHub repo visibility is intentionally excluded. Flip via Settings →
General → Change visibility (5 seconds in the web UI).

Run from repo root:
    python -m pytest tests/test_flip_visibility.py -v
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

# Canonical SolarHive HF repo registry.
# Keep this list sorted by (repo_type, repo_id) for stable diffs.
REPOS: List[Tuple[str, str]] = [
    ("Truthseeker87/solarhive-community-solar-multimodal", "dataset"),
    ("Truthseeker87/solarhive-26b-a4b-lora", "model"),
    ("Truthseeker87/solarhive-26b-a4b-merged", "model"),
    ("Truthseeker87/solarhive-26b-a4b-nf4", "model"),
    ("Truthseeker87/solarhive-e4b-gguf", "model"),
    ("Truthseeker87/solarhive-e4b-ollama", "model"),
    ("Truthseeker87/solarhive", "space"),
]


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="flip_visibility",
        description="Flip SolarHive HF repo visibility between public and private.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--make",
        choices=["public", "private"],
        help="Target visibility for all SolarHive HF repos.",
    )
    g.add_argument(
        "--verify-only",
        action="store_true",
        help="Print current visibility state and exit without changing anything.",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually call the HF API. Without this flag, the script is a dry run.",
    )
    return p.parse_args(argv)


def get_state(api, repo_id, repo_type):
    """Return True if the repo is currently private."""
    info = api.repo_info(repo_id, repo_type=repo_type)
    return bool(info.private)


def set_visibility(api, repo_id, repo_type, private):
    """Update the repo's visibility. Returns the API response."""
    return api.update_repo_settings(
        repo_id=repo_id, repo_type=repo_type, private=private
    )


def plan(repos, target_private):
    """Pure function: compute (to_change, no_op) given a target visibility.

    Args:
        repos: list of (repo_id, repo_type, current_private) tuples.
        target_private: bool — True for private, False for public.

    Returns:
        (to_change, no_op) — two lists of (repo_id, repo_type) tuples.
    """
    to_change = []
    no_op = []
    for repo_id, repo_type, current_private in repos:
        if current_private == target_private:
            no_op.append((repo_id, repo_type))
        else:
            to_change.append((repo_id, repo_type))
    return to_change, no_op


def fmt_table(rows, headers=("repo_id", "type", "private")):
    """Format a list of (repo_id, repo_type, is_private) rows as a fixed-width table."""
    widths = [
        max(len(headers[i]), max((len(str(r[i])) for r in rows), default=0))
        for i in range(len(headers))
    ]
    line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    out = [line, sep]
    for r in rows:
        out.append("  ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)


def main(argv=None, api_factory=None):
    """Entry point. Returns exit code (0 = success, non-zero = error).

    api_factory: optional callable returning an HfApi instance.
        Injected for testability; defaults to constructing HfApi() with HF_TOKEN.
    """
    args = parse_args(argv if argv is not None else sys.argv[1:])

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        return 2

    if api_factory is None:
        from huggingface_hub import HfApi
        api_factory = lambda: HfApi(token=token)
    api = api_factory()

    # Fetch current state for all repos.
    current = []
    for repo_id, repo_type in REPOS:
        try:
            is_private = get_state(api, repo_id, repo_type)
            current.append((repo_id, repo_type, is_private))
        except Exception as e:
            print(f"ERROR reading {repo_id} ({repo_type}): {e}", file=sys.stderr)
            return 3

    if args.verify_only:
        print("=== Current visibility ===")
        print(fmt_table(current))
        return 0

    target_private = args.make == "private"
    to_change, no_op = plan(current, target_private)

    print(f"=== Plan: make all {args.make} ===")
    print(f"  to change: {len(to_change)} repo(s)")
    for rid, rt in to_change:
        print(f"    {rid}  ({rt})")
    print(f"  already {args.make}: {len(no_op)} repo(s)")

    if not args.execute:
        print("\nDRY RUN — no changes made. Re-run with --execute to apply.")
        return 0

    if not to_change:
        print(f"\nAll repos are already {args.make}. Nothing to do.")
        return 0

    print("\n=== Executing ===")
    failures = []
    for rid, rt in to_change:
        try:
            set_visibility(api, rid, rt, target_private)
            print(f"  OK  {rid}  ({rt})")
        except Exception as e:
            print(f"  FAIL  {rid}  ({rt})  {e}", file=sys.stderr)
            failures.append((rid, rt, str(e)))

    if failures:
        print(f"\n{len(failures)} repo(s) failed to update.", file=sys.stderr)
        return 4
    print(f"\nAll {len(to_change)} repo(s) flipped to {args.make}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
