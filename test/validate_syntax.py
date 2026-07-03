#!/usr/bin/env python3
"""Quick syntax validation for modified files."""

import py_compile
import sys

files_to_check = [
    "CODI/core/improver.py",
    "CODI/core/executor.py",
    "CODI/core/validator.py",
    "CODI/state/temp_db.py",
    "CODI/agent.py",
]

errors = []
for filepath in files_to_check:
    try:
        py_compile.compile(filepath, doraise=True)
        print(f"✓ {filepath}")
    except Exception as e:
        print(f"✗ {filepath}: {e}")
        errors.append((filepath, e))

if errors:
    print(f"\n{len(errors)} file(s) had errors")
    sys.exit(1)
else:
    print(f"\n✓ All {len(files_to_check)} files compiled successfully")
    sys.exit(0)
