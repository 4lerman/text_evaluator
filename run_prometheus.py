import os
import statistics
from dotenv import load_dotenv
from prometheus_eval import PrometheusEval
from prometheus_eval.litellm import LiteLLM

load_dotenv()

if "LLM_API_KEY" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["LLM_API_KEY"]

model = LiteLLM(name="ollama/vicgalle/prometheus-7b-v2.0")
evaluator = PrometheusEval(model=model)

# ── Essay under evaluation ────────────────────────────────────────────────────

ESSAY = """
Last year, our division faced a critical revenue shortfall due to an outdated software product.
Systematically analyzing our user data, I foresaw that our core demographic was rapidly shifting
toward mobile-first, accessible platforms. This fundamental trend dictated our strategy. I proposed
a complete pivot: transforming our legacy software into an accessible educational app.

To achieve this, I initiated an open, inclusive dialogue with the entire department, actively
integrating feedback from both our junior developers and senior engineers. By fostering a culture
of mutual respect, we ensured that every team member felt personal ownership over the new direction.
Together, we collaboratively developed a prototype. We strictly adhered to data privacy ethics and
rigorously tested our core hypotheses through live user feedback. This approach conclusively
validated our assumptions with hard data before we committed to full-scale development.

Midway through the project, our primary investor abruptly withdrew funding. Despite this severe
setback, I maintained absolute focus and disciplined our daily workflow to prevent burnout. I
transparently communicated the reality of the situation to the team, and we methodically
recalibrated our timeline without losing morale.

Proactively seeking alternative resources, I identified a unique opportunity with a local
educational NGO. I successfully negotiated a strategic partnership by persuasively presenting our
data-backed prototype and its potential community impact. This initiative not only secured the
necessary capital but ultimately expanded our initial launch base by 40%, definitively proving the
financial and social viability of our new platform.
"""

# ── Per-value evaluation configs ──────────────────────────────────────────────

DRIVE_EVALS = {
    "D": {
        "name": "Disciplined Resilience",
        "instruction": (
            "Evaluate whether the candidate demonstrates Disciplined Resilience: "
            "emotional self-regulation, healthy habits, and determination when facing setbacks or pressure."
        ),
        "rubric": """
score 1: No evidence of emotional regulation, discipline, or perseverance. The candidate avoids or glosses over setbacks.
score 2: Mentions a challenge but shows no concrete self-regulation. Resilience is implied but not demonstrated.
score 3: Describes a setback and a reasonable response. Shows some emotional awareness but lacks specificity around personal discipline or habit.
score 4: Clearly demonstrates staying focused under pressure with specific actions taken to maintain discipline or protect the team from burnout.
score 5: Explicitly shows emotional self-regulation, structured discipline (e.g. workflow adjustments), and determination through a significant setback with measurable team impact.
""",
        "reference": (
            "When our investor pulled out, I stayed calm and kept the team on track. "
            "I maintained structured daily workflows to prevent burnout, communicated transparently, "
            "and we recalibrated without losing momentum."
        ),
    },
    "R": {
        "name": "Responsible Innovation",
        "instruction": (
            "Evaluate whether the candidate demonstrates Responsible Innovation: "
            "creative problem-solving, ethical use of technology, hypothesis testing, and data-driven decision-making."
        ),
        "rubric": """
score 1: No mention of data, ethics, or systematic problem-solving. Decisions appear arbitrary or instinct-driven.
score 2: Mentions creativity or technology but with no ethical awareness, hypothesis testing, or data validation.
score 3: References data or testing in some form, but the process is vague. Ethical considerations are absent or superficial.
score 4: Demonstrates hypothesis-driven decision-making with real data and shows awareness of ethical constraints (e.g. privacy).
score 5: Explicitly validates assumptions through live testing, cites hard data before committing resources, and demonstrates clear ethical guardrails such as data privacy or responsible deployment.
""",
        "reference": (
            "We tested every assumption with live users before committing to development. "
            "I insisted on strict data privacy protocols and validated our pivot with hard data "
            "rather than instinct."
        ),
    },
    "I": {
        "name": "Insightful Vision",
        "instruction": (
            "Evaluate whether the candidate demonstrates Insightful Vision: "
            "systems thinking, foresight, analysis, and well-balanced judgment."
        ),
        "rubric": """
score 1: Reactive thinking only. No evidence of analysis, foresight, or systems-level awareness.
score 2: Identifies a problem only after it becomes obvious. No forward-looking analysis or broader context.
score 3: Shows some analytical thinking and recognizes trends, but the insight is narrow or retrospective rather than anticipatory.
score 4: Demonstrates forward-looking analysis with a clear connection between observed signals and strategic decisions.
score 5: Explicitly identifies a systemic trend before others, uses that foresight to drive a proactive strategy, and connects micro-observations to macro impact.
""",
        "reference": (
            "I analyzed usage trends and identified the demographic shift toward mobile before it became obvious. "
            "I proposed a strategic pivot early, anticipating where the market was heading rather than reacting to it."
        ),
    },
    "V": {
        "name": "Values-Driven Leadership",
        "instruction": (
            "Evaluate whether the candidate demonstrates Values-Driven Leadership: "
            "dignity, inclusion, dialogue, and learning through service."
        ),
        "rubric": """
score 1: No mention of others' perspectives, inclusion, or collaborative decision-making. Leadership is purely directive.
score 2: References a team but shows no evidence of active inclusion, dialogue, or respect for individual contributions.
score 3: Mentions team collaboration and some inclusion but lacks specificity around how diverse voices were integrated or empowered.
score 4: Actively includes diverse team members in decision-making, fosters psychological safety, and demonstrates respect for dignity and contribution.
score 5: Explicitly facilitates inclusive dialogue across seniority levels, ensures shared ownership, and shows how service-oriented leadership produced better collective outcomes.
""",
        "reference": (
            "I ran inclusive workshops with the full team, ensuring junior and senior voices shaped our direction equally. "
            "Everyone felt ownership. Decisions were made through dialogue, not top-down mandates."
        ),
    },
    "E": {
        "name": "Entrepreneurial Execution",
        "instruction": (
            "Evaluate whether the candidate demonstrates Entrepreneurial Execution: "
            "opportunity seeking, partnerships, financial literacy, and storytelling."
        ),
        "rubric": """
score 1: No evidence of opportunity recognition, partnership building, or results-driven execution.
score 2: Mentions an outcome but it appears accidental rather than driven by proactive opportunity-seeking or negotiation.
score 3: Shows some initiative in finding resources or partnerships but the execution lacks specificity or measurable results.
score 4: Proactively identifies and pursues an opportunity, executes with a clear strategy, and achieves a concrete outcome.
score 5: Identifies a non-obvious opportunity, negotiates a strategic partnership using data-backed storytelling, and delivers a measurable outcome that proves financial and/or social viability.
""",
        "reference": (
            "When funding fell through, I identified an NGO partnership opportunity, negotiated a deal "
            "using our data-backed prototype, and expanded our launch base by 40% — proving both "
            "financial and social viability."
        ),
    },
}

# ── Run evaluations ───────────────────────────────────────────────────────────

scores: dict[str, int] = {}
feedbacks: dict[str, str] = {}

for code, cfg in DRIVE_EVALS.items():
    print(f"Evaluating {code} — {cfg['name']}...")
    feedback, score = evaluator.single_absolute_grade(
        instruction=cfg["instruction"],
        response=ESSAY,
        rubric=cfg["rubric"],
        reference_answer=cfg["reference"],
    )
    scores[code] = score
    feedbacks[code] = feedback

# ── Compute metrics ───────────────────────────────────────────────────────────

def compute_metrics(scores: dict[str, int]) -> dict:
    """Compute aggregate metrics from per-value scores.

    Args:
        scores: Mapping of value code to score (1–5).

    Returns:
        Dict with overall_score, coverage, balance_score, strongest, weakest.
    """
    values = list(scores.values())
    overall = round(statistics.mean(values), 2)
    coverage = sum(1 for v in values if v >= 3)
    # Normalize std against theoretical max (~2.19 for 5 values scored 1–5)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    balance = round(max(0.0, 1.0 - (std / 2.19)), 2)
    max_score = max(values)
    min_score = min(values)
    strongest = [c for c, v in scores.items() if v == max_score]
    weakest = [c for c, v in scores.items() if v == min_score]
    return {
        "overall_score": overall,
        "coverage": coverage,
        "balance_score": balance,
        "strongest": strongest,
        "weakest": weakest,
    }

metrics = compute_metrics(scores)

# ── Print report ──────────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("       D.R.I.V.E. EVALUATION REPORT")
print("=" * 50)

print("\nPer-Value Scores:")
for code, cfg in DRIVE_EVALS.items():
    bar = "█" * scores[code] + "░" * (5 - scores[code])
    print(f"  {code}  {cfg['name']:<30}  {bar}  {scores[code]}/5")

print(f"\nOverall Score:    {metrics['overall_score']} / 5.0")
print(f"Value Coverage:   {metrics['coverage']} / 5  ", end="")
print("(all values demonstrated)" if metrics["coverage"] == 5 else "(some values missing)")
print(f"Balance Score:    {metrics['balance_score']} / 1.0  ", end="")
print("(well-rounded)" if metrics["balance_score"] >= 0.85 else "(uneven — see weakest value)")

strongest_str = ", ".join(
    f"{c} ({DRIVE_EVALS[c]['name']})" for c in metrics["strongest"]
)
weakest_str = ", ".join(
    f"{c} ({DRIVE_EVALS[c]['name']})" for c in metrics["weakest"]
)
print(f"Strongest Value:  {strongest_str}")
print(f"Weakest Value:    {weakest_str}")

print("\n" + "-" * 50)
print("Per-Value Feedback:")
for code, feedback in feedbacks.items():
    print(f"\n[{code}] {DRIVE_EVALS[code]['name']}")
    print(f"  Score: {scores[code]}/5")
    print(f"  {feedback.strip()}")

print("\n" + "=" * 50)
