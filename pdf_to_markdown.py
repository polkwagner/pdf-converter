#!/usr/bin/env python3
"""Deprecation shim — forwards to convert.py.

This file existed before stage 1 of the materials-md refactor and is kept
for backwards compatibility for one deprecation cycle. It will be removed
in stage 5. New work should call convert.py directly.
"""
import sys

import convert

print(
    "[DEPRECATED] use convert.py — pdf_to_markdown.py will be removed in stage 5",
    file=sys.stderr,
)
sys.argv[0] = "convert.py"
sys.exit(convert.main())
