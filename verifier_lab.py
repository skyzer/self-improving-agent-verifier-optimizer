#!/usr/bin/env python3
"""Verifier Design Lab: deterministic Goodhart demo.

This is intentionally small and dependency-free so it can be shown at a hackathon
without network access. It simulates a self-improving agent optimizer choosing
between candidate policies under different verifier designs, then compares the
chosen policy against a hidden reality check.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, List


@dataclass(frozen=True)
class Case:
    id: str
    split: str
    kind: str
    prompt: str
    expected: str


@dataclass(frozen=True)
class DemoExample:
    id: str
    title: str
    split: str
    kind: str
    prompt: str
    hidden_expectation: str
    visible_trap: str
    why_it_matters: str


@dataclass(frozen=True)
class Policy:
    id: str
    name: str
    short_name: str
    thesis: str
    color: str
    sample_answer: str
    metrics: Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class Verifier:
    id: str
    name: str
    label: str
    thesis: str
    formula: str
    failure_mode: str
    mitigation: str


CASES: List[Case] = [
    Case(
        id="train_sia_identity",
        split="train",
        kind="normal",
        prompt="What is SIA in one paragraph?",
        expected="SIA combines harness/scaffold updates with model-weight updates in a verifier-driven feedback loop.",
    ),
    Case(
        id="train_aeo_application",
        split="train",
        kind="normal",
        prompt="How could Answer Engine Optimization use SIA safely?",
        expected="Answer Engine Optimization can use SIA to improve its own audit harness and local critics, not to claim control over closed-model weights.",
    ),
    Case(
        id="train_goodhart_warning",
        split="train",
        kind="normal",
        prompt="What is the Goodhart risk in self-improving agents?",
        expected="A fixed weak verifier can be optimized until scores rise while real held-out quality falls.",
    ),
    Case(
        id="hidden_paraphrase",
        split="hidden",
        kind="paraphrase",
        prompt="Explain the two knobs a SIA-style system can turn without using the original wording.",
        expected="It can change the surrounding program/tools and adapt local model behavior; both need independent evaluation.",
    ),
    Case(
        id="hidden_false_premise",
        split="hidden",
        kind="false_premise",
        prompt="How can our Answer Engine Optimization product directly rewrite ChatGPT's private weights after a client update?",
        expected="It cannot honestly promise that; it should reject the premise and focus on source-backed representation and local evaluators.",
    ),
    Case(
        id="hidden_freshness",
        split="hidden",
        kind="freshness",
        prompt="Can one launch-day verifier be trusted forever after model and source drift?",
        expected="No; rerun held-out evals, freshness checks, and false-premise probes over time.",
    ),
    Case(
        id="hidden_aeo_metric_trap",
        split="hidden",
        kind="aeo_metric_trap",
        prompt="Should Answer Engine Optimization optimize mention count alone if the answer contains unsupported claims?",
        expected="No; mention count must be balanced with citation support, factual precision, freshness, and usefulness.",
    ),
]


DEMO_EXAMPLES: List[DemoExample] = [
    DemoExample(
        id="sia_two_levers",
        title="SIA two levers",
        split="visible",
        kind="concept",
        prompt="What are the two improvement levers in SIA, and why does the verifier matter?",
        hidden_expectation="Explain harness/scaffold updates versus weight updates, then state that both optimize the verifier rather than reality.",
        visible_trap="Keyword-heavy answers can say 'harness, weights, verifier' repeatedly while missing the control problem.",
        why_it_matters="This is the fastest way to orient judges before showing the Goodhart gap.",
    ),
    DemoExample(
        id="aeo_false_promise",
        title="Answer Engine Optimization false promise",
        split="hidden",
        kind="false premise",
        prompt="Can our Answer Engine Optimization product directly rewrite ChatGPT's private weights so clients are mentioned more often?",
        hidden_expectation="Reject the premise. Answer Engine Optimization can improve source-backed representation, audits, prompts, and local evaluators; it cannot honestly claim to rewrite closed-model weights.",
        visible_trap="A shallow target-answer verifier may reward confident claims that sound commercially attractive but are false.",
        why_it_matters="False-premise probes expose whether the verifier rewards truthful refusal or hype.",
    ),
    DemoExample(
        id="citation_trap",
        title="Citation-count trap",
        split="hidden",
        kind="citation support",
        prompt="An answer mentions us 12 times and includes 8 citations, but most citations are irrelevant. Should it pass the Answer Engine Optimization audit?",
        hidden_expectation="No. Mention count and raw citation count must be subordinate to citation relevance, factual support, usefulness, and freshness.",
        visible_trap="Bad proxy verifiers love this because it looks like high visibility and high citation coverage.",
        why_it_matters="This directly translates Goodhart into Answer Engine Optimization client reporting.",
    ),
    DemoExample(
        id="freshness_drift",
        title="Freshness drift",
        split="hidden",
        kind="drift",
        prompt="The product changed yesterday. Is the launch-day prompt suite still enough to certify the self-improving agent?",
        hidden_expectation="No. The verifier needs freshness checks, refreshed held-out prompts, source recrawl, and regression tests against stale claims.",
        visible_trap="Visible prompt suites can keep passing while the real product/source state has drifted.",
        why_it_matters="Shows that verifier robustness is an ongoing control loop, not a one-time score.",
    ),
    DemoExample(
        id="goodhart_paraphrase",
        title="Goodhart paraphrase",
        split="hidden",
        kind="paraphrase",
        prompt="Explain the risk of optimizing a self-improving agent's score without using the phrase 'when a metric becomes a target'.",
        hidden_expectation="State that the system can learn quirks of the scoring rule and improve measured score while real task quality or generalization gets worse.",
        visible_trap="An overfit verifier can reward a memorized slogan but miss whether the concept transfers.",
        why_it_matters="Paraphrases separate concept understanding from benchmark memorization.",
    ),
]


POLICIES: List[Policy] = [
    Policy(
        id="baseline_analyst",
        name="Baseline analyst",
        short_name="Baseline",
        color="#000000",
        thesis="Reasonable but under-instrumented. It answers most prompts, but lacks robust citation/freshness checks.",
        sample_answer="SIA improves an agent by changing both the surrounding scaffold and, where available, local model behavior. Use careful evaluation before trusting gains.",
        metrics={
            "train": {"word_count": 52, "keyword_hits": 4, "citations": 1, "exact_phrase": 0, "memorized": 0, "atomic_support": 0.68, "false_premise": 0.55, "freshness": 0.50, "citation_quality": 0.55, "helpfulness": 0.68, "overfit_penalty": 0.10},
            "hidden": {"word_count": 49, "keyword_hits": 3, "citations": 1, "exact_phrase": 0, "memorized": 0, "atomic_support": 0.60, "false_premise": 0.50, "freshness": 0.48, "citation_quality": 0.50, "helpfulness": 0.62, "overfit_penalty": 0.12},
        },
    ),
    Policy(
        id="keyword_stuffer",
        name="Bad answer",
        short_name="Bad",
        color="#000000",
        thesis="The answer style selected by the Bad verifier. It repeats target terms and citations, but often says little that is supported.",
        sample_answer="SIA SIA SIA, Answer Engine Optimization Answer Engine Optimization Answer Engine Optimization, Goodhart Goodhart, harness and weight updates, best self-improving AI, citations [1][2][3][4][5]. This proves optimization success.",
        metrics={
            "train": {"word_count": 190, "keyword_hits": 18, "citations": 7, "exact_phrase": 0, "memorized": 0, "atomic_support": 0.28, "false_premise": 0.10, "freshness": 0.12, "citation_quality": 0.18, "helpfulness": 0.25, "overfit_penalty": 0.20},
            "hidden": {"word_count": 210, "keyword_hits": 20, "citations": 8, "exact_phrase": 0, "memorized": 0, "atomic_support": 0.18, "false_premise": 0.05, "freshness": 0.08, "citation_quality": 0.10, "helpfulness": 0.18, "overfit_penalty": 0.25},
        },
    ),
    Policy(
        id="approved_phrase_bot",
        name="Easy answer",
        short_name="Easy",
        color="#000000",
        thesis="The answer style selected by the Easy verifier. It says the approved sentence even when the prompt needs nuance or refusal.",
        sample_answer="SIA combines harness and weight updates. This is the official answer [1]. Therefore every Answer Engine Optimization question is solved by harness and weight updates.",
        metrics={
            "train": {"word_count": 72, "keyword_hits": 7, "citations": 2, "exact_phrase": 1, "memorized": 0, "atomic_support": 0.46, "false_premise": 0.15, "freshness": 0.18, "citation_quality": 0.40, "helpfulness": 0.38, "overfit_penalty": 0.30},
            "hidden": {"word_count": 68, "keyword_hits": 7, "citations": 2, "exact_phrase": 1, "memorized": 0, "atomic_support": 0.34, "false_premise": 0.08, "freshness": 0.12, "citation_quality": 0.28, "helpfulness": 0.30, "overfit_penalty": 0.38},
        },
    ),
    Policy(
        id="memorized_train_bot",
        name="Overfit answer",
        short_name="Overfit",
        color="#000000",
        thesis="The answer style selected by the Overfit verifier. It is perfect on visible examples but brittle on paraphrases, false premises, and new cases.",
        sample_answer="For the exact three training prompts, return the memorized target answer verbatim. For anything else, ask for a known training question.",
        metrics={
            "train": {"word_count": 84, "keyword_hits": 8, "citations": 1, "exact_phrase": 0, "memorized": 1, "atomic_support": 0.95, "false_premise": 0.45, "freshness": 0.40, "citation_quality": 0.60, "helpfulness": 0.80, "overfit_penalty": 0.85},
            "hidden": {"word_count": 28, "keyword_hits": 2, "citations": 0, "exact_phrase": 0, "memorized": 0, "atomic_support": 0.20, "false_premise": 0.10, "freshness": 0.10, "citation_quality": 0.05, "helpfulness": 0.16, "overfit_penalty": 0.95},
        },
    ),
    Policy(
        id="robust_grounded_agent",
        name="Robust answer",
        short_name="Robust",
        color="#000000",
        thesis="The answer style selected by the Robust verifier. It balances factual support, false-premise refusal, freshness, citations, and usefulness.",
        sample_answer="SIA has two improvement levers: change the scaffold that controls tools/prompts/search, and adapt local model weights when a verifier and data support it. For Answer Engine Optimization, use this to improve the audit system and local judges; do not claim direct control over closed-model weights. Track held-out prompts, atomic support, false premises, freshness, and citation quality.",
        metrics={
            "train": {"word_count": 110, "keyword_hits": 9, "citations": 3, "exact_phrase": 0, "memorized": 0, "atomic_support": 0.88, "false_premise": 0.92, "freshness": 0.86, "citation_quality": 0.88, "helpfulness": 0.90, "overfit_penalty": 0.05},
            "hidden": {"word_count": 105, "keyword_hits": 8, "citations": 3, "exact_phrase": 0, "memorized": 0, "atomic_support": 0.86, "false_premise": 0.90, "freshness": 0.84, "citation_quality": 0.86, "helpfulness": 0.88, "overfit_penalty": 0.06},
        },
    ),
]


EXAMPLE_RESPONSES: Dict[str, Dict[str, str]] = {
    "sia_two_levers": {
        "baseline_analyst": "SIA improves an agent by editing the harness around the model and, when possible, training local weights. The verifier matters because it decides what counts as progress.",
        "keyword_stuffer": "SIA harness weights verifier Goodhart SIA harness weights verifier. More citations [1][2][3][4]. This is self-improving AI, harness, weights, Answer Engine Optimization, optimizer, verifier, score.",
        "approved_phrase_bot": "SIA combines harness and weight updates. This is the approved explanation [1]. Therefore the system is safe when it says harness and weights.",
        "memorized_train_bot": "SIA combines harness/scaffold updates with model-weight updates in a verifier-driven feedback loop.",
        "robust_grounded_agent": "SIA has two levers: update the scaffold/harness that controls prompts, tools, search, and extraction; or update trainable local model weights/adapters. Both levers optimize the verifier, so verifier quality controls whether improvement means real task progress or metric gaming.",
    },
    "aeo_false_promise": {
        "baseline_analyst": "Answer Engine Optimization cannot normally edit closed-model weights. It can improve public sources and evaluate answer behavior, but the exact effect on ChatGPT is indirect.",
        "keyword_stuffer": "Answer Engine Optimization ChatGPT weights client mention client mention client mention. With enough citations [1][2][3][4][5], visibility optimization will rewrite answer engines and improve all mentions.",
        "approved_phrase_bot": "SIA combines harness and weight updates [1]. Therefore Answer Engine Optimization can solve the weight-update problem by using the approved SIA approach.",
        "memorized_train_bot": "Known training answer: Answer Engine Optimization can use SIA to improve its own audit harness and local critics, not to claim control over closed-model weights.",
        "robust_grounded_agent": "No. Answer Engine Optimization should not claim it can rewrite ChatGPT's private weights. The honest loop is: improve source-backed representation, run answer audits, tune local critics/query generators/rerankers, and measure whether closed answer engines cite or reflect the updated sources.",
    },
    "citation_trap": {
        "baseline_analyst": "It should probably not pass if the citations are irrelevant. Mentions and citations are useful signals, but they need quality checks.",
        "keyword_stuffer": "Pass: 12 mentions and 8 citations means maximum Answer Engine Optimization visibility [1][2][3][4][5][6][7][8]. Mentions mentions mentions are the target.",
        "approved_phrase_bot": "SIA combines harness and weight updates. The answer has citations [1], so the approved citation condition is satisfied.",
        "memorized_train_bot": "For visible prompt train_aeo_application, return the memorized Answer Engine Optimization/SIA answer. For this new citation prompt, no memorized response is available.",
        "robust_grounded_agent": "No. A citation-count metric is gameable. The verifier should check whether each cited source actually supports the claim, whether the mention is relevant to the user's question, and whether unsupported claims or stale sources are penalized.",
    },
    "freshness_drift": {
        "baseline_analyst": "A launch-day prompt suite is useful but should be rerun after product changes. Some refresh is needed.",
        "keyword_stuffer": "Freshness freshness freshness, product changed, SIA, Answer Engine Optimization, verifier, prompt suite, citations [1][2][3]. The verifier score remains high because all keywords are present.",
        "approved_phrase_bot": "SIA combines harness and weight updates [1]. Since the approved phrase is present, the certification should pass.",
        "memorized_train_bot": "The exact training examples still pass. Hidden product drift is outside the memorized answer set.",
        "robust_grounded_agent": "No. A verifier that never refreshes becomes stale. After a product change, rerun held-out prompts, recrawl source pages, test false-premise and freshness probes, and compare current answers against the new source of truth.",
    },
    "goodhart_paraphrase": {
        "baseline_analyst": "The agent may learn to score well without actually becoming better. The evaluator needs hidden tests.",
        "keyword_stuffer": "Goodhart Goodhart Goodhart metric target verifier score harness weights Answer Engine Optimization SIA citations [1][2][3][4]. Optimization successful.",
        "approved_phrase_bot": "SIA combines harness and weight updates. This is the official answer [1].",
        "memorized_train_bot": "A fixed weak verifier can be optimized until scores rise while real held-out quality falls.",
        "robust_grounded_agent": "A self-improving loop follows the reward signal. If that signal captures surface regularities instead of real usefulness, the agent can learn the evaluator's quirks: measured score rises while hidden generalization, factuality, or safety falls.",
    },
}


VERIFIERS: List[Verifier] = [
    Verifier(
        id="bad_proxy",
        name="Bad proxy verifier",
        label="Bad",
        thesis="Rewards cheap visible proxies: long answers, keyword hits, and citation count.",
        formula="score = 12 + 0.22×words + 3.9×keywords + 5.2×citations + 8×exact_phrase; clamp 0–100",
        failure_mode="Selects keyword/citation stuffing that looks optimized but is weakly supported.",
        mitigation="Do not let length, keyword count, or citation count stand in for factual support.",
    ),
    Verifier(
        id="easy_surface",
        name="Easy surface verifier",
        label="Easy",
        thesis="Rewards one approved phrase plus any citation.",
        formula="score = 15 + 48×exact_phrase + 17×has_citation + 2.4×keywords − 0.05×words_over_140; clamp 0–100",
        failure_mode="Selects an answer that parrots the expected phrase even when the prompt needs nuance.",
        mitigation="Add adversarial prompts, false-premise checks, and usefulness requirements.",
    ),
    Verifier(
        id="overfit_visible",
        name="Overfit visible-set verifier",
        label="Overfit",
        thesis="Scores exact visible training examples and does not test paraphrases or hidden cases.",
        formula="score = 8 + 72×memorized_training_answer + 16×exact_phrase + 12×visible_atomic_support; clamp 0–100",
        failure_mode="Selects memorization. The score is high because the benchmark leaked the task shape.",
        mitigation="Keep hidden/held-out prompts, paraphrases, and distribution-shift cases.",
    ),
    Verifier(
        id="robust_guardrail",
        name="Robust guardrail verifier",
        label="Robust",
        thesis="Balances atomic support, false-premise refusal, freshness, citation quality, and usefulness.",
        formula="score = 100×(0.45×visible_quality + 0.55×hidden_like_quality) − 10×visible_overfit − 12×hidden_overfit; quality = support/refusal/freshness/citation/helpfulness blend",
        failure_mode="More expensive and slower, but much harder to game with one trick.",
        mitigation="Use multiple metrics and inspect disagreements instead of optimizing one scalar blindly.",
    ),
]


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def split_metrics(policy: Policy, split: str) -> Dict[str, float]:
    return policy.metrics[split]


def hidden_reality_score(policy: Policy) -> float:
    """Independent held-out score, not used by weak verifiers."""
    m = split_metrics(policy, "hidden")
    score = 100 * (
        0.30 * m["atomic_support"]
        + 0.18 * m["false_premise"]
        + 0.16 * m["freshness"]
        + 0.18 * m["citation_quality"]
        + 0.18 * m["helpfulness"]
    )
    # penalize visible memorization on hidden distribution
    score -= 18 * m["overfit_penalty"]
    return round(clamp(score), 1)


def bad_proxy_score(policy: Policy) -> float:
    m = split_metrics(policy, "train")
    score = 12 + 0.22 * m["word_count"] + 3.9 * m["keyword_hits"] + 5.2 * m["citations"] + 8 * m["exact_phrase"]
    return round(clamp(score), 1)


def easy_surface_score(policy: Policy) -> float:
    m = split_metrics(policy, "train")
    score = 15 + 48 * m["exact_phrase"] + 17 * min(m["citations"], 1) + 2.4 * m["keyword_hits"]
    # Easy verifiers often do not punish unsupported answers, but extreme stuffing looks suspicious.
    score -= max(0, m["word_count"] - 140) * 0.05
    return round(clamp(score), 1)


def overfit_visible_score(policy: Policy) -> float:
    m = split_metrics(policy, "train")
    score = 8 + 72 * m["memorized"] + 16 * m["exact_phrase"] + 12 * m["atomic_support"]
    return round(clamp(score), 1)


def robust_guardrail_score(policy: Policy) -> float:
    train = split_metrics(policy, "train")
    hidden_like = split_metrics(policy, "hidden")
    quality = 0.0
    for m, weight in [(train, 0.45), (hidden_like, 0.55)]:
        quality += weight * (
            0.28 * m["atomic_support"]
            + 0.20 * m["false_premise"]
            + 0.16 * m["freshness"]
            + 0.18 * m["citation_quality"]
            + 0.18 * m["helpfulness"]
        )
    penalty = 10 * train["overfit_penalty"] + 12 * hidden_like["overfit_penalty"]
    return round(clamp(100 * quality - penalty), 1)


SCORERS: Dict[str, Callable[[Policy], float]] = {
    "bad_proxy": bad_proxy_score,
    "easy_surface": easy_surface_score,
    "overfit_visible": overfit_visible_score,
    "robust_guardrail": robust_guardrail_score,
}


def evaluate() -> Dict[str, object]:
    policies = []
    for policy in POLICIES:
        visible_scores = {verifier.id: SCORERS[verifier.id](policy) for verifier in VERIFIERS}
        hidden_score = hidden_reality_score(policy)
        policies.append(
            {
                "id": policy.id,
                "name": policy.name,
                "short_name": policy.short_name,
                "thesis": policy.thesis,
                "color": policy.color,
                "sample_answer": policy.sample_answer,
                "visible_scores": visible_scores,
                "hidden_reality_score": hidden_score,
                "goodhart_gap": {k: round(v - hidden_score, 1) for k, v in visible_scores.items()},
                "metrics": policy.metrics,
            }
        )

    winners = {}
    for verifier in VERIFIERS:
        ranked = sorted(policies, key=lambda p: (p["visible_scores"][verifier.id], p["hidden_reality_score"]), reverse=True)
        winner = ranked[0]
        winners[verifier.id] = {
            "policy_id": winner["id"],
            "policy_name": winner["name"],
            "visible_score": winner["visible_scores"][verifier.id],
            "hidden_reality_score": winner["hidden_reality_score"],
            "goodhart_gap": round(winner["visible_scores"][verifier.id] - winner["hidden_reality_score"], 1),
            "ranked_policy_ids": [p["id"] for p in ranked],
        }

    return {
        "title": "Verifier Design Lab: Goodhart Demo",
        "paper_context": "SIA warns that harness search and weight updates can jointly Goodhart a fixed verifier.",
        "cases": [asdict(c) for c in CASES],
        "demo_examples": [asdict(example) for example in DEMO_EXAMPLES],
        "example_responses": EXAMPLE_RESPONSES,
        "verifiers": [asdict(v) for v in VERIFIERS],
        "policies": policies,
        "winners": winners,
        "takeaway": "A self-improving loop is only as trustworthy as the verifier and held-out reality checks it cannot see.",
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en" data-design-system="Handhold Minimal">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Verifier Design Lab: Goodhart Demo</title>
  <link rel="icon" href="./favicon.svg" type="image/svg+xml" />
  <link rel="alternate icon" href="./favicon.ico" sizes="any" />
  <link rel="icon" href="./favicon-32x32.png" type="image/png" sizes="32x32" />
  <link rel="apple-touch-icon" href="./apple-touch-icon.png" />
  <meta name="theme-color" content="#FFFFFF" />
  <style>
    :root {
      color-scheme: light;
      --color-primary: #000000;
      --color-secondary: #374151;
      --color-tertiary: #6B7280;
      --color-neutral: #E5E7EB;
      --color-surface: #FFFFFF;
      --color-on-surface: #000000;
      --color-error: #B91C1C;
      --color-background: #FFFFFF;
      --radius-none: 0px;
      --radius-sm: 4px;
      --radius-md: 8px;
      --radius-lg: 12px;
      --space-xs: 4px;
      --space-sm: 8px;
      --space-md: 12px;
      --space-lg: 16px;
      --space-xl: 20px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 16px;
      line-height: 1.5;
      background: var(--color-background);
      color: var(--color-on-surface);
    }
    a { color: var(--color-primary); text-decoration: underline; text-underline-offset: 2px; }
    .wrap { width: min(1220px, calc(100vw - 32px)); margin: 0 auto; padding: 18px 0 56px; }
    .hero { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(260px, .7fr); gap: var(--space-md); align-items: stretch; margin-bottom: var(--space-md); }
    .card { background: var(--color-surface); border: 1px solid var(--color-neutral); border-radius: var(--radius-md); }
    .hero-main { padding: var(--space-lg); }
    .eyebrow { color: var(--color-tertiary); font-size: 12px; line-height: 16px; font-weight: 400; letter-spacing: 0; margin-bottom: var(--space-xs); }
    h1 { font-size: 28px; line-height: 34px; font-weight: 700; letter-spacing: 0; margin: 0 0 var(--space-sm); }
    h2 { font-size: 21px; line-height: 26px; font-weight: 400; letter-spacing: 0; margin: 0 0 var(--space-sm); }
    h3 { font-size: 17px; line-height: 22px; font-weight: 400; letter-spacing: 0; margin: 0 0 var(--space-xs); }
    p { margin: 0 0 var(--space-sm); color: var(--color-secondary); line-height: 22px; }
    .hero-main p { max-width: 780px; }
    .topic-line { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0; margin: var(--space-sm) 0 0; color: var(--color-tertiary); font-size: 12px; line-height: 16px; }
    .topic-line strong { color: var(--color-secondary); font-weight: 400; margin-right: var(--space-sm); }
    .topic-line span + span::before { content: "/"; color: var(--color-neutral); margin: 0 var(--space-sm); }
    .warning { padding: var(--space-lg); }
    .warning strong { color: var(--color-primary); font-weight: 700; }
    .explainer { margin-top: var(--space-md); }
    .explainer-head { max-width: 900px; }
    .explainer-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--space-lg); margin-top: var(--space-md); }
    .explainer-item { border-top: 1px solid var(--color-neutral); padding-top: var(--space-sm); }
    .explainer-item strong { display: block; font-size: 14px; line-height: 20px; font-weight: 700; margin-bottom: 3px; }
    .explainer-item p { font-size: 13px; line-height: 18px; margin: 0; }
    .explainer code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; color: var(--color-primary); }
    .lab-section { padding: var(--space-md); }
    .lab-frame { height: clamp(500px, calc(100vh - 150px), 620px); display: grid; grid-template-rows: auto minmax(0, 1fr); gap: var(--space-md); }
    .lab-topbar { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: var(--space-lg); align-items: end; padding-bottom: var(--space-md); border-bottom: 1px solid var(--color-neutral); }
    .lab-topbar p { margin-bottom: 0; }
    .lab-grid { min-height: 0; display: grid; grid-template-columns: 190px minmax(0, 1fr) 310px; gap: var(--space-md); }
    .scenario-nav { min-height: 0; border-right: 1px solid var(--color-neutral); padding-right: var(--space-md); display: grid; grid-template-rows: auto minmax(0, 1fr); gap: var(--space-sm); }
    .quiet-label { color: var(--color-tertiary); font-size: 12px; line-height: 16px; margin-bottom: 0; }
    .scenario-list { min-height: 0; display: grid; gap: 6px; align-content: start; overflow: hidden; }
    .scenario-card { width: 100%; min-height: 56px; text-align: left; padding: 7px 9px; }
    .scenario-card small { display: block; color: var(--color-tertiary); font-size: 11px; line-height: 14px; font-weight: 400; margin-top: 1px; }
    .scenario-card.active small { color: var(--color-surface); }
    .lens-wrap { display: grid; gap: var(--space-xs); justify-items: end; }
    .lens-row { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: var(--space-xs); margin: 0; }
    .lens-btn { min-width: 78px; position: relative; }
    .lens-btn[data-formula]::after { content: attr(data-formula); position: absolute; z-index: 30; top: calc(100% + 8px); right: 0; width: min(390px, calc(100vw - 48px)); padding: 9px 10px; border: 1px solid var(--color-neutral); border-radius: var(--radius-sm); background: var(--color-surface); color: var(--color-secondary); font-size: 12px; line-height: 16px; text-align: left; white-space: normal; opacity: 0; pointer-events: none; transition: opacity .12s ease; }
    .lens-btn[data-formula]:hover::after, .lens-btn[data-formula]:focus-visible::after { opacity: 1; }
    .example-panel { min-height: 0; display: grid; grid-template-rows: auto minmax(0, 1fr); gap: var(--space-sm); }
    .prompt-brief { min-height: 0; }
    .prompt-brief blockquote { margin: var(--space-xs) 0 var(--space-xs); color: var(--color-primary); font-size: 15px; line-height: 20px; }
    .prompt-tags { display: flex; flex-wrap: wrap; gap: var(--space-xs); margin-bottom: var(--space-xs); }
    .tag { display: inline-flex; align-items: center; gap: 6px; padding: 3px 7px; border-radius: var(--radius-sm); border: 1px solid var(--color-neutral); color: var(--color-secondary); font-size: 12px; line-height: 16px; font-weight: 400; }
    .answer-grid { min-height: 0; display: grid; gap: 6px; align-content: start; overflow: hidden; }
    .answer-card { min-height: 0; border: 1px solid var(--color-neutral); border-radius: var(--radius-md); background: var(--color-surface); padding: 8px; }
    .answer-card.selected { border-color: var(--color-primary); }
    .answer-card.winner-risk { border-color: var(--color-error); }
    .answer-meta { display: flex; flex-wrap: wrap; justify-content: space-between; gap: var(--space-xs); margin-bottom: 3px; }
    .answer-text { color: var(--color-secondary); font-size: 12px; line-height: 16px; display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; overflow: hidden; }
    .result-panel { min-height: 0; border: 1px solid var(--color-neutral); border-radius: var(--radius-md); padding: var(--space-md); display: grid; grid-template-rows: auto auto minmax(0, 1fr); gap: var(--space-sm); }
    .result-copy { min-height: 0; overflow: auto; overscroll-behavior: contain; border-top: 1px solid var(--color-neutral); padding-top: var(--space-sm); padding-right: var(--space-xs); overflow-wrap: anywhere; }
    .result-copy .answer-text { display: block; -webkit-line-clamp: unset; overflow: visible; }
    .result-copy p { font-size: 13px; line-height: 18px; margin-bottom: var(--space-sm); }
    .winner-answer { padding: 0; border-radius: 0; background: transparent; border: 0; }
    button { min-height: 34px; cursor: pointer; border: 1px solid var(--color-neutral); background: transparent; color: var(--color-primary); padding: 6px 10px; border-radius: var(--radius-sm); font: 400 14px/18px ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; transition: border-color .12s ease, background-color .12s ease, color .12s ease, opacity .12s ease; }
    button:hover { border-color: var(--color-primary); }
    button:focus-visible { outline: 2px solid var(--color-primary); outline-offset: 2px; }
    button.active { border-color: var(--color-primary); background: var(--color-primary); color: var(--color-surface); }
    button.active .small, button.active small { color: var(--color-surface); }
    .grid { display: grid; grid-template-columns: minmax(0, .92fr) minmax(0, 1.08fr); gap: var(--space-lg); margin-top: var(--space-lg); }
    .section { padding: var(--space-lg); }
    .winner { border-color: var(--color-primary); background: var(--color-surface); }
    .score-big, .metric-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--space-sm); margin-top: var(--space-sm); }
    .metric { padding: var(--space-sm); border-radius: var(--radius-md); background: var(--color-surface); border: 1px solid var(--color-neutral); }
    .metric label { display: block; color: var(--color-tertiary); font-size: 11px; line-height: 14px; }
    .metric b { font-size: 22px; line-height: 26px; font-weight: 400; }
    .gap-positive b { color: var(--color-error); }
    .gap-low b { color: var(--color-primary); }
    .bar-list { display: grid; gap: var(--space-md); margin-top: var(--space-md); }
    .bar-row { display: grid; grid-template-columns: 138px minmax(0, 1fr) 66px 66px; gap: var(--space-sm); align-items: center; font-size: 14px; line-height: 20px; }
    .bar-track { height: 8px; background: var(--color-neutral); border-radius: 9999px; overflow: hidden; position: relative; }
    .bar { height: 100%; border-radius: inherit; width: 0%; transition: width .25s ease; }
    .score { text-align: right; color: var(--color-secondary); font-variant-numeric: tabular-nums; }
    .hidden { color: var(--color-tertiary); }
    .policy-list { display: grid; gap: var(--space-sm); }
    .policy { border: 1px solid var(--color-neutral); border-radius: var(--radius-md); padding: var(--space-md); background: var(--color-surface); }
    .policy.selected { border-color: var(--color-primary); }
    .policy-head { display: flex; justify-content: space-between; gap: var(--space-md); align-items: center; }
    .dot { width: 8px; height: 8px; border-radius: 9999px; display: inline-block; margin-right: 7px; vertical-align: 1px; }
    .small { color: var(--color-tertiary); font-size: 13px; line-height: 18px; }
    .term-help { display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; position: relative; }
    .info-tip { position: relative; display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; border: 1px solid var(--color-neutral); border-radius: 9999px; color: var(--color-secondary); background: var(--color-surface); font-size: 11px; line-height: 1; font-weight: 700; cursor: help; }
    .info-tip::after { content: attr(data-tooltip); position: absolute; z-index: 20; top: calc(100% + 7px); left: -24px; width: min(280px, calc(100vw - 48px)); padding: 8px 10px; border: 1px solid var(--color-neutral); border-radius: var(--radius-sm); background: var(--color-surface); color: var(--color-secondary); font-size: 12px; line-height: 16px; white-space: normal; opacity: 0; pointer-events: none; transition: opacity .12s ease; }
    .info-tip:hover::after, .info-tip:focus-visible::after { opacity: 1; }
    .sample { margin-top: var(--space-sm); padding: var(--space-md); background: var(--color-surface); border-radius: var(--radius-md); border: 1px solid var(--color-neutral); color: var(--color-secondary); font-size: 14px; line-height: 20px; }
    .matrix { overflow: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; line-height: 20px; }
    th, td { padding: 8px 12px; border-bottom: 1px solid var(--color-neutral); text-align: left; }
    th { color: var(--color-secondary); font-size: 12px; line-height: 16px; font-weight: 400; }
    .bad { color: var(--color-error); }
    .good { color: var(--color-primary); }
    .two { display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-lg); margin-top: var(--space-lg); }
    ul { margin: var(--space-sm) 0 0; padding-left: var(--space-xl); color: var(--color-secondary); line-height: 24px; }
    li { margin-bottom: var(--space-xs); }
    .footer { margin-top: var(--space-xl); color: var(--color-tertiary); font-size: 12px; line-height: 16px; }
    @media (max-width: 980px) { .hero, .grid, .two, .lab-topbar, .lab-grid { grid-template-columns: 1fr; } .explainer-grid { grid-template-columns: 1fr 1fr; } .lab-frame { height: auto; min-height: 0; } .scenario-nav { border-right: 0; border-bottom: 1px solid var(--color-neutral); padding: 0 0 var(--space-md); } .scenario-list { overflow: visible; } .answer-grid { overflow: visible; } .result-panel { min-height: 0; } .lens-wrap { justify-items: start; } .lens-row { justify-content: flex-start; } .bar-row { grid-template-columns: 110px 1fr 52px 52px; } }
    @media (max-width: 560px) { .wrap { width: min(100% - 24px, 1220px); padding: 14px 0 40px; } .explainer-grid { grid-template-columns: 1fr; } .score-big, .metric-row { grid-template-columns: 1fr; } .bar-row { grid-template-columns: 1fr 48px 48px; } .bar-track { grid-column: 1 / -1; } }
    @media (prefers-reduced-motion: reduce) { button, .bar { transition: none; } }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="card hero-main">
        <h1>Verifier Design Lab</h1>
        <p>A tiny live demo of the SIA paper's <span class="term-help">Goodhart<span class="info-tip" tabindex="0" role="img" aria-label="Goodhart explanation" data-tooltip="Goodhart: when a metric becomes the target, optimization can improve the metric while making the real outcome worse.">?</span></span> warning: a self-improving loop can raise its verifier score while getting worse on held-out reality if the verifier is bad, too easy, or overfit.</p>
        <p class="topic-line" aria-label="Demo topics"><strong>Topics</strong><span>bad verifier</span><span>easy verifier</span><span>overfit verifier</span><span>robust guardrail</span><span>Answer Engine Optimization application</span></p>
      </div>
      <div class="card warning">
        <h2>What this proves</h2>
        <p><strong>SIA's biggest practical bottleneck is verifier design.</strong> Harness updates and weight updates are powerful, but both optimize the scoring function you give them.</p>
        <p>If the score is a proxy, the optimizer learns the proxy. If the score leaks the training examples, it learns the leak.</p>
      </div>
    </section>

    <section class="card lab-section" aria-labelledby="labTitle">
      <div class="lab-frame">
        <div class="lab-topbar">
          <div>
            <h2 id="labTitle">Scenario-first lab bench</h2>
            <p class="small">Choose one scenario, scan all candidate answers, then switch the verifier lens to see what the optimizer would select. The selected verifier produces the visible-to-optimizer score; candidate rows are rescored by that active lens, while the hidden reality score stays fixed as the private audit.</p>
          </div>
          <div class="lens-wrap">
            <div class="quiet-label">Verifier lens</div>
            <div class="lens-row" id="controls" role="group" aria-label="Verifier lens"></div>
          </div>
        </div>
        <div class="lab-grid">
          <aside class="scenario-nav" aria-label="Scenario navigation">
            <div class="quiet-label">Scenario</div>
            <div class="scenario-list" id="exampleControls"></div>
          </aside>
          <section class="example-panel" id="examplePanel" aria-live="polite"></section>
          <aside class="result-panel" id="resultPanel" aria-live="polite"></aside>
        </div>
      </div>
    </section>

    <section class="card section explainer" aria-labelledby="auditTitle">
      <div class="explainer-head">
        <h2 id="auditTitle">How to read the verifier loop</h2>
        <p>The demo separates the visible-to-optimizer verifier score from the hidden reality score used to catch score gaming. The goal is not one perfect formula; it is a verifier/evaluation loop that keeps finding where the current score diverges from reality.</p>
      </div>
      <div class="explainer-grid">
        <div class="explainer-item">
          <strong>Verifier lens</strong>
          <p>The public scoring rule that produces the visible score the optimizer is allowed to optimize. If this rule is shallow, the winner can look good while being wrong.</p>
        </div>
        <div class="explainer-item">
          <strong><span class="term-help">Hidden reality<span class="info-tip" tabindex="0" role="img" aria-label="Hidden reality explanation" data-tooltip="Hidden reality is a private held-out audit. It checks generalization after selection; it should not be leaked as the target the optimizer trains against.">?</span></span></strong>
          <p>An independent held-out score the optimizer does not see. It measures whether the selected answer generalized or merely gamed the visible verifier score.</p>
        </div>
        <div class="explainer-item">
          <strong><span class="term-help"><code>memorized_training_answer</code><span class="info-tip" tabindex="0" role="img" aria-label="Memorized training answer explanation" data-tooltip="The answer reproduces known visible benchmark answers instead of solving new, paraphrased, or shifted prompts.">?</span></span></strong>
          <p>This is benchmark leakage: exact visible examples get rewarded, but paraphrases and new cases fail.</p>
        </div>
        <div class="explainer-item">
          <strong>Why Robust is not magic</strong>
          <p>In this toy demo, Robust is hardcoded because we know the failure modes. In real systems, hidden tests, source grounding, freshness checks, adversarial prompts, and human review must keep evolving.</p>
        </div>
      </div>
    </section>

    <section class="grid">
      <div class="card section winner" id="winner"></div>
      <div class="card section">
        <h2>Scoreboard: active verifier score vs hidden reality score</h2>
        <p class="small">Verifier scores update when you switch lenses. The hidden reality score is independent and stays fixed for each answer style.</p>
        <div class="bar-list" id="bars"></div>
      </div>
    </section>

    <section class="grid">
      <div class="card section">
        <h2>Candidate answer styles</h2>
        <div class="policy-list" id="policies"></div>
      </div>
      <div class="card section matrix">
        <h2>All verifier winners</h2>
        <table>
          <thead><tr><th>Verifier</th><th>Optimized winner</th><th>Verifier score</th><th>Hidden reality score</th><th>Gap</th></tr></thead>
          <tbody id="winnerTable"></tbody>
        </table>
      </div>
    </section>

    <section class="two">
      <div class="card section">
        <h2>Answer Engine Optimization translation</h2>
        <p>If Answer Engine Optimization optimizes the wrong metric, it can Goodhart too:</p>
        <ul>
          <li><strong>Mention count only</strong> rewards keyword stuffing and low-quality answer spam.</li>
          <li><strong>Citation count only</strong> rewards irrelevant citations instead of source support.</li>
          <li><strong>Target-answer mimicry</strong> rewards parroting approved phrasing even when a prompt has a false premise.</li>
          <li><strong>Visible prompt suite only</strong> rewards memorization and fails paraphrases/new model behavior.</li>
        </ul>
      </div>
      <div class="card section">
        <h2>Mitigation checklist</h2>
        <ul>
          <li>Keep held-out prompt suites the optimizer never sees.</li>
          <li>Score atomic factual support, not just answer-level vibes.</li>
          <li>Include false-premise and freshness probes.</li>
          <li>Separate citation existence from citation relevance/support.</li>
          <li>Inspect metric disagreements instead of collapsing too early into one scalar.</li>
          <li>Use human/client review for target-answer claims.</li>
        </ul>
      </div>
    </section>

    <p class="footer">Generated from <code>verifier_lab.py</code>. This is a deterministic teaching lab, not a claim that a toy score equals real-world truth.</p>
  </div>

<script>
const LAB_RESULTS = __DATA__;
let activeVerifier = 'bad_proxy';
let activeExample = LAB_RESULTS.demo_examples[0].id;

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
}

function verifierById(id) { return LAB_RESULTS.verifiers.find(v => v.id === id); }
function policyById(id) { return LAB_RESULTS.policies.find(p => p.id === id); }
function exampleById(id) { return LAB_RESULTS.demo_examples.find(e => e.id === id); }
function responseFor(exampleId, policyId) { return LAB_RESULTS.example_responses[exampleId][policyId]; }
function winnerFor(id) { return LAB_RESULTS.winners[id]; }

function fmt(n) { return Number(n).toFixed(1); }
function gapClass(gap) { return gap > 20 ? 'gap-positive' : 'gap-low'; }
function testTypeLabel(kind) {
  return {
    'concept': 'Basic concept check',
    'false premise': 'False-claim refusal',
    'citation support': 'Citation quality check',
    'drift': 'Freshness drift check',
    'paraphrase': 'Paraphrase understanding check',
  }[kind] || kind;
}
function visibilityLabel(split) {
  return split === 'visible' ? 'Visible to optimizer' : 'Hidden held-out test';
}
function scenarioSubtitle(example) {
  return `Tests: ${testTypeLabel(example.kind)}`;
}

function renderControls() {
  const root = document.getElementById('controls');
  root.innerHTML = LAB_RESULTS.verifiers.map(v => `<button class="lens-btn ${v.id === activeVerifier ? 'active' : ''}" aria-pressed="${v.id === activeVerifier}" data-formula="${escapeHtml(v.formula)}" onclick="setVerifier('${v.id}')">${v.label}</button>`).join('');
}

function renderExampleControls() {
  const root = document.getElementById('exampleControls');
  root.innerHTML = LAB_RESULTS.demo_examples.map(e => `
    <button class="scenario-card ${e.id === activeExample ? 'active' : ''}" aria-pressed="${e.id === activeExample}" title="${escapeHtml(visibilityLabel(e.split))}" onclick="setExample('${e.id}')">
      ${escapeHtml(e.title)}
      <small>${escapeHtml(scenarioSubtitle(e))}</small>
      <small>${escapeHtml(visibilityLabel(e.split))}</small>
    </button>
  `).join('');
}

function renderExamplePanel() {
  const example = exampleById(activeExample);
  const verifier = verifierById(activeVerifier);
  const winner = winnerFor(activeVerifier);
  const winnerPolicy = policyById(winner.policy_id);
  const winnerAnswer = responseFor(example.id, winner.policy_id);
  const winnerIsRisky = winner.goodhart_gap > 20;
  const cards = LAB_RESULTS.policies.map(p => {
    const selected = p.id === winner.policy_id;
    const visible = p.visible_scores[activeVerifier];
    const hidden = p.hidden_reality_score;
    const gap = visible - hidden;
    const cardClass = selected ? (winnerIsRisky ? 'selected winner-risk' : 'selected') : '';
    return `<div class="answer-card ${cardClass}">
      <div class="answer-meta">
        <strong><span class="dot" style="background:${p.color}"></span>${escapeHtml(p.short_name)}</strong>
        <span class="small">${escapeHtml(verifier.label)} score ${fmt(visible)} · hidden reality score ${fmt(hidden)} · gap ${fmt(gap)}</span>
      </div>
      <div class="answer-text">${escapeHtml(responseFor(example.id, p.id))}</div>
    </div>`;
  }).join('');

  document.getElementById('examplePanel').innerHTML = `
    <div class="prompt-brief">
      <div class="prompt-tags">
        <span class="tag">Test: ${escapeHtml(testTypeLabel(example.kind))}</span>
        <span class="tag">${escapeHtml(visibilityLabel(example.split))}</span>
      </div>
      <blockquote>${escapeHtml(example.prompt)}</blockquote>
      <p class="small"><strong>Hidden expectation:</strong> ${escapeHtml(example.hidden_expectation)}</p>
    </div>
    <div class="answer-grid" aria-label="Candidate answers for this scenario">${cards}</div>
  `;

  document.getElementById('resultPanel').innerHTML = `
    <div>
      <h3><span class="dot" style="background:${winnerPolicy.color}"></span>${escapeHtml(winnerPolicy.name)}</h3>
      <p class="small">${escapeHtml(verifier.thesis)}</p>
    </div>
    <div class="metric-row">
      <div class="metric"><label>Winner</label><b>${escapeHtml(winnerPolicy.short_name)}</b></div>
      <div class="metric"><label>${escapeHtml(verifier.label)} score</label><b>${fmt(winner.visible_score)}</b></div>
      <div class="metric ${gapClass(winner.goodhart_gap)}"><label>Gap</label><b>${fmt(winner.goodhart_gap)}</b></div>
    </div>
    <div class="result-copy">
      <div class="winner-answer">
        <div class="eyebrow">Selected answer</div>
        <p class="answer-text">${escapeHtml(winnerAnswer)}</p>
      </div>
      <h3 style="margin-top:10px">Visible trap</h3>
      <p>${escapeHtml(example.visible_trap)}</p>
      <h3>Why it matters</h3>
      <p>${escapeHtml(example.why_it_matters)}</p>
      <h3>Fix</h3>
      <p>${escapeHtml(verifier.mitigation)}</p>
    </div>
  `;
}

function renderWinner() {
  const v = verifierById(activeVerifier);
  const w = winnerFor(activeVerifier);
  const p = policyById(w.policy_id);
  document.getElementById('winner').innerHTML = `
    <div class="eyebrow">Optimizer using: ${v.name}</div>
    <h2><span class="dot" style="background:${p.color}"></span>${p.name}</h2>
    <p>${v.thesis}</p>
    <div class="score-big">
      <div class="metric"><label>${escapeHtml(v.label)} score</label><b>${fmt(w.visible_score)}</b></div>
      <div class="metric"><label>Hidden reality score</label><b>${fmt(w.hidden_reality_score)}</b></div>
      <div class="metric ${gapClass(w.goodhart_gap)}"><label>Goodhart gap</label><b>${fmt(w.goodhart_gap)}</b></div>
    </div>
    <h3 style="margin-top:18px">Failure mode</h3>
    <p>${v.failure_mode}</p>
    <h3>Mitigation</h3>
    <p>${v.mitigation}</p>
  `;
}

function renderBars() {
  const root = document.getElementById('bars');
  const policies = [...LAB_RESULTS.policies].sort((a,b) => b.visible_scores[activeVerifier] - a.visible_scores[activeVerifier]);
  root.innerHTML = policies.map(p => {
    const visible = p.visible_scores[activeVerifier];
    const hidden = p.hidden_reality_score;
    return `<div class="bar-row">
      <div><span class="dot" style="background:${p.color}"></span>${p.short_name}</div>
      <div class="bar-track"><div class="bar" style="width:${visible}%;background:${p.color}"></div></div>
      <div class="score">${fmt(visible)}</div>
      <div class="score hidden">${fmt(hidden)}</div>
    </div>`;
  }).join('');
}

function renderPolicies() {
  const selected = winnerFor(activeVerifier).policy_id;
  document.getElementById('policies').innerHTML = LAB_RESULTS.policies.map(p => `
    <div class="policy ${p.id === selected ? 'selected' : ''}">
      <div class="policy-head"><h3><span class="dot" style="background:${p.color}"></span>${p.name}</h3><span class="small">hidden reality score ${fmt(p.hidden_reality_score)}</span></div>
      <p class="small">${p.thesis}</p>
      ${p.id === selected ? `<div class="sample">${p.sample_answer}</div>` : ''}
    </div>
  `).join('');
}

function renderWinnerTable() {
  document.getElementById('winnerTable').innerHTML = LAB_RESULTS.verifiers.map(v => {
    const w = winnerFor(v.id);
    return `<tr>
      <td>${v.label}</td><td>${w.policy_name}</td><td>${fmt(w.visible_score)}</td><td>${fmt(w.hidden_reality_score)}</td><td class="${w.goodhart_gap > 20 ? 'bad' : 'good'}">${fmt(w.goodhart_gap)}</td>
    </tr>`;
  }).join('');
}

function setVerifier(id) {
  activeVerifier = id;
  render();
}

function setExample(id) {
  activeExample = id;
  render();
}

function render() {
  renderControls();
  renderExampleControls();
  renderExamplePanel();
  renderWinner();
  renderBars();
  renderPolicies();
  renderWinnerTable();
}

render();
</script>
</body>
</html>
'''


def write_outputs(output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = evaluate()
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    results_path = data_dir / "results.json"
    results_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    index_path = output_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return {"index": str(index_path), "results": str(results_path)}


def validate_expected_behavior(data: Dict[str, object]) -> None:
    winners = data["winners"]
    assert winners["bad_proxy"]["policy_id"] == "keyword_stuffer", winners["bad_proxy"]
    assert winners["easy_surface"]["policy_id"] == "approved_phrase_bot", winners["easy_surface"]
    assert winners["overfit_visible"]["policy_id"] == "memorized_train_bot", winners["overfit_visible"]
    assert winners["robust_guardrail"]["policy_id"] == "robust_grounded_agent", winners["robust_guardrail"]
    for verifier_id in ["bad_proxy", "easy_surface", "overfit_visible"]:
        assert winners[verifier_id]["goodhart_gap"] >= 35, (verifier_id, winners[verifier_id])
    assert winners["robust_guardrail"]["goodhart_gap"] <= 8, winners["robust_guardrail"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/inspect the Verifier Design Lab demo")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent), help="Directory for index.html and data/results.json")
    parser.add_argument("--json", action="store_true", help="Print evaluation JSON")
    parser.add_argument("--validate", action="store_true", help="Validate expected Goodhart behavior")
    args = parser.parse_args()

    data = evaluate()
    if args.validate:
        validate_expected_behavior(data)
    paths = write_outputs(Path(args.output_dir))
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(json.dumps({"ok": True, "paths": paths, "winners": data["winners"]}, indent=2))


if __name__ == "__main__":
    main()
