# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal JSONC (JSON + // / /* */ comments) loader."""

from __future__ import annotations

import json
import re
from pathlib import Path


def strip_jsonc_comments(text: str) -> str:
    """Remove // line comments and /* block comments */, keeping strings intact."""
    out = []
    i = 0
    n = len(text)
    in_string = False
    escape = False
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "/":
                i += 2
                while i < n and text[i] not in "\r\n":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i = min(n, i + 2)
                continue
        out.append(c)
        i += 1
    # Trailing commas before } or ] are common in hand-edited jsonc.
    cleaned = re.sub(r",(\s*[}\]])", r"\1", "".join(out))
    return cleaned


def load_jsonc(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(strip_jsonc_comments(text))
