#!/usr/bin/env python3
"""Apply the audited loopback-bind change to Firecrawl v1.7.0 Playwright."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, original: str, patched: str) -> str:
    if patched in text:
        return text
    if text.count(original) != 1:
        raise RuntimeError(f"expected exactly one upstream fragment: {original!r}")
    return text.replace(original, patched, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    text = args.source.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "const port = process.env.PORT || 3003;",
        "const port = process.env.PORT || 3003;\nconst host = process.env.HOST || '127.0.0.1';",
    )
    text = replace_once(
        text,
        "app.listen(port, () => {\n  initializeBrowser().then(() => {\n"
        "    console.log(`Server is running on port ${port}`);",
        "app.listen(Number(port), host, () => {\n  initializeBrowser().then(() => {\n"
        "    console.log(`Server is running on ${host}:${port}`);",
    )
    temporary = args.source.with_name(f".{args.source.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(args.source)


if __name__ == "__main__":
    main()
