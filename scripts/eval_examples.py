#!/usr/bin/env python3
"""
eval_examples.py — Run all examples from drive_examples.json against the
D.R.I.V.E. Evaluator API and report pass/fail against expected_confirms bounds.

Usage:
    python scripts/eval_examples.py
    python scripts/eval_examples.py --url http://localhost:8000 --timeout 90
    python scripts/eval_examples.py --category negative
    python scripts/eval_examples.py --difficulty hard
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

EXAMPLES_PATH = Path(__file__).parent.parent / "examples" / "drive_examples.json"


@dataclass
class EvalResult:
    id: str
    category: str
    difficulty: str
    primary_value: str
    expected_min: int
    expected_max: int
    actual_confirms: int = -1
    confirmed_values: list[str] = field(default_factory=list)
    passed: bool = False
    failure_reason: str = ""
    error: str = ""
    notes: str = ""
    all_results: list[dict] = field(default_factory=list)


def check(example: dict, confirms: int, confirmed_values: list[str]) -> tuple[bool, str]:
    """Return (passed, failure_reason)."""
    lo = example["expected_confirms"]["min"]
    hi = example["expected_confirms"]["max"]

    if confirms < lo:
        return False, f"got {confirms} confirm(s), expected ≥{lo}"
    if confirms > hi:
        return False, f"got {confirms} confirm(s), expected ≤{hi}"

    # For single-value positive examples, verify the primary value appears
    pv = example.get("primary_value", "none")
    if example["category"] == "positive" and pv not in ("none", "multiple"):
        if pv not in confirmed_values:
            return False, f"value '{pv}' not in confirmed set {confirmed_values}"

    return True, ""


async def eval_one(
    client: httpx.AsyncClient,
    example: dict,
    base_url: str,
    timeout: float,
) -> EvalResult:
    lo = example["expected_confirms"]["min"]
    hi = example["expected_confirms"]["max"]
    base = EvalResult(
        id=example["id"],
        category=example["category"],
        difficulty=example["difficulty"],
        primary_value=example.get("primary_value", "none"),
        expected_min=lo,
        expected_max=hi,
        notes=example.get("notes", ""),
    )

    try:
        resp = await client.post(
            f"{base_url}/evaluate",
            json={"text": example["text"]},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        confirms = len(results)
        confirmed_values = sorted({r["value_code"] for r in results})
        passed, reason = check(example, confirms, confirmed_values)
        base.actual_confirms = confirms
        base.confirmed_values = confirmed_values
        base.passed = passed
        base.failure_reason = reason
        base.all_results = results
    except httpx.ConnectError:
        base.actual_confirms = -1
        base.error = f"Connection refused — is the server running at {base_url}?"
    except httpx.TimeoutException:
        base.actual_confirms = -1
        base.error = f"Request timed out after {timeout}s"
    except Exception as e:
        base.actual_confirms = -1
        base.error = str(e)

    return base


def fmt_range(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else f"{lo}–{hi}"


def category_color(cat: str) -> str:
    return {
        "positive": GREEN,
        "negative": RED,
        "ambiguous": YELLOW,
    }.get(cat, RESET)


async def run(base_url: str, timeout: float, filter_cat: str | None, filter_diff: str | None) -> int:
    with open(EXAMPLES_PATH) as f:
        data = json.load(f)

    examples = data["examples"]
    if filter_cat:
        examples = [e for e in examples if e["category"] == filter_cat]
    if filter_diff:
        examples = [e for e in examples if e["difficulty"] == filter_diff]

    if not examples:
        print(f"{RED}No examples match the given filters.{RESET}")
        return 1

    print(f"\n{BOLD}D.R.I.V.E. Evaluator — Example Eval{RESET}")
    print(f"  Server   : {CYAN}{base_url}{RESET}")
    print(f"  Examples : {len(examples)}")
    print(f"  Timeout  : {timeout}s per request\n")

    col = f"{'ID':<18} {'CAT':<11} {'DIFF':<8} {'EXPECT':<8} {'GOT':<5} {'VALUES':<16} STATUS"
    print(col)
    print("─" * 82)

    results: list[EvalResult] = []

    async with httpx.AsyncClient() as client:
        for ex in examples:
            r = await eval_one(client, ex, base_url, timeout)
            results.append(r)

            cat_c = category_color(r.category)
            status = f"{GREEN}{BOLD}PASS{RESET}" if r.passed else f"{RED}{BOLD}FAIL{RESET}"
            values_str = ", ".join(r.confirmed_values) if r.confirmed_values else "—"
            got_str = "ERR" if r.actual_confirms == -1 else str(r.actual_confirms)

            print(
                f"{r.id:<18} {cat_c}{r.category:<11}{RESET} {r.difficulty:<8} "
                f"{fmt_range(r.expected_min, r.expected_max):<8} {got_str:<5} {values_str:<16} {status}"
            )

            if r.error:
                print(f"  {RED}↳ {r.error}{RESET}")
            elif not r.passed and r.failure_reason:
                print(f"  {YELLOW}↳ {r.failure_reason}{RESET}")
                if r.notes:
                    print(f"  {DIM}  note: {r.notes[:110]}{'...' if len(r.notes) > 110 else ''}{RESET}")

            # Print highlights for confirmed values
            for res in r.all_results:
                v_code = res["value_code"]
                reason = res["reasoning"]
                text = res["text"]
                # Find the evidence highlight (pos_category="EVIDENCE_QUOTE")
                ev_quotes = [h["token"] for h in res.get("highlights", []) if h["pos_category"] == "EVIDENCE_QUOTE"]
                
                print(f"  {CYAN}→ {BOLD}{v_code}{RESET}: {reason}")
                
                display_text = text
                for q in ev_quotes:
                    # Very simple ANSI bolding for the quote
                    display_text = display_text.replace(q, f"{BOLD}{YELLOW}{q}{RESET}{DIM}")
                
                print(f"    {DIM}\"{display_text}\"{RESET}")

    # ── Summary ──────────────────────────────────────────────────────────────

    total = len(results)
    n_pass = sum(1 for r in results if r.passed)
    n_fail = total - n_pass
    pct = int(100 * n_pass / total) if total else 0

    print("\n" + "─" * 82)
    print(f"\n{BOLD}Summary{RESET}  {n_pass}/{total} passed ({pct}%)")

    print(f"\n  {'Category':<14} {'Pass':<6} {'Total'}")
    for cat in ("positive", "negative", "ambiguous"):
        sub = [r for r in results if r.category == cat]
        if not sub:
            continue
        sp = sum(1 for r in sub if r.passed)
        bar = ("█" * sp) + ("░" * (len(sub) - sp))
        print(f"  {category_color(cat)}{cat:<14}{RESET} {sp:<6} {len(sub)}  {bar}")

    print(f"\n  {'Difficulty':<14} {'Pass':<6} {'Total'}")
    for diff in ("easy", "medium", "hard"):
        sub = [r for r in results if r.difficulty == diff]
        if not sub:
            continue
        sp = sum(1 for r in sub if r.passed)
        bar = ("█" * sp) + ("░" * (len(sub) - sp))
        print(f"  {diff:<14} {sp:<6} {len(sub)}  {bar}")

    if n_fail:
        print(f"\n{RED}Failed:{RESET}")
        for r in results:
            if not r.passed:
                detail = r.error or r.failure_reason
                print(f"  • {r.id:<18} {detail}")

    print()
    return 0 if n_fail == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval drive_examples.json against the D.R.I.V.E. Evaluator API"
    )
    parser.add_argument(
        "--url", default="http://localhost:8000",
        help="Base URL of the evaluator API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Per-request timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--category", choices=["positive", "negative", "ambiguous"],
        help="Run only examples in this category"
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard"],
        help="Run only examples with this difficulty"
    )
    args = parser.parse_args()

    code = asyncio.run(run(args.url, args.timeout, args.category, args.difficulty))
    sys.exit(code)


if __name__ == "__main__":
    main()
