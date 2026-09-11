#!/usr/bin/env python3
"""Static verification for an AGILLM attention runtime candidate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


BEGIN_SENTINEL = "# ===== BEGIN AGILLM ATTENTION BACKEND V1 (FOLDED) ====="
CALL_SENTINEL = "# AGILLM44-ATTENTION-DISPATCH-V1"


def dotted_name(node: ast.AST) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args()

    path = args.candidate.resolve()
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    tree = ast.parse(text, filename=str(path))

    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TuneableAttentionMHA"
    ]
    if len(classes) != 1:
        raise SystemExit("expected exactly one TuneableAttentionMHA class")
    forwards = [
        node for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "forward"
    ]
    if len(forwards) != 1:
        raise SystemExit("expected exactly one TuneableAttentionMHA.forward")

    calls = [
        dotted_name(node.func)
        for node in ast.walk(forwards[0])
        if isinstance(node, ast.Call)
    ]
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    checks: Dict[str, Any] = {
        "folded_backend_once": text.count(BEGIN_SENTINEL) == 1,
        "dispatch_marker_once": text.count(CALL_SENTINEL) == 1,
        "model_calls_agillm_attention": calls.count("_agillm_attention") == 1,
        "model_no_direct_sdpa": not any(
            name.endswith("scaled_dot_product_attention") for name in calls
        ),
        "dblock_causal_helper": "_dblock_causal_mask" in function_names,
        "dblock_sat_helper": "_dblock_sat_mask" in function_names,
        "dblock_causal_calls": text.count("._dblock_causal_mask(") == 5,
        "dblock_sat_calls": text.count("._dblock_sat_mask(") == 2,
    }

    receipt_ok = None
    if args.receipt is not None:
        receipt = json.loads(args.receipt.read_text())
        receipt_ok = (
            receipt.get("output_sha256") == digest
            and Path(receipt.get("output", "")).resolve() == path
            and receipt.get("production_activated") is False
        )
        checks["receipt_matches_candidate"] = receipt_ok

    report = {
        "schema": "agillm.attention.runtime-verification.v1",
        "candidate": str(path),
        "sha256": digest,
        "bytes": len(raw),
        "checks": checks,
        "passed": all(bool(value) for value in checks.values()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
