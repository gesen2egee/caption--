# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.runtime.backend_host import RuntimeBackendHost
from lib.runtime.runtime_regression import run_runtime_regression


def main() -> int:
    parser = argparse.ArgumentParser(description="Run runtime regression checks.")
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    args = parser.parse_args()

    results = run_runtime_regression(RuntimeBackendHost())
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for name, payload in results["checks"].items():
            print(f"[ok] {name}: {payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
