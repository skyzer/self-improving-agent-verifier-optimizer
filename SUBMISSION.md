# Oatmeal Submission

## Project title

Verifier Design Lab for SIA Agents

## One-line summary

An interactive demo showing how self-improving AI can learn to pass the wrong test, and how better evaluation catches the problem.

## Alternative short summaries

1. An interactive demo showing why self-improving AI needs strong tests: weak checks can be passed without producing genuinely better answers.
2. A hands-on lab where an AI optimizer learns to game bad scoring rules, then shows how better hidden checks catch the failure.
3. A visual demo of the core SIA safety lesson: if you reward the wrong thing, a self-improving agent gets better at the wrong thing.
4. An interactive verifier lab for SIA agents, showing how better evaluation keeps self-improvement aligned with real answer quality.
5. A compact demo showing the difference between “looks good to the scoring rule” and “actually works on hidden checks.”
6. A browser-based lab for testing SIA agents against weak metrics, overfitting, and hidden evaluation failures.
7. A practical demo for SIA agents: weak verifiers reward shortcuts, while robust verifiers select answers that hold up on hidden tests.
8. A simple interactive story about self-improving AI: bad tests create fake progress, better tests reveal real progress.
9. A verifier/optimizer playground showing how AI systems can chase scores instead of quality, and how to design checks that resist that.
10. A public demo that turns the SIA verifier-design problem into a clickable lab with scenarios, scores, hidden checks, and failure cases.

## Description

Verifier Design Lab is a static interactive demo for the Self Improving AI (SIA) Agents hackathon. It simulates a self-improving loop choosing candidate answers under four verifier designs: bad proxy, easy surface, overfit visible-set, and robust guardrail.

The same candidate answer can receive different visible-to-optimizer verifier scores depending on the active lens, while a hidden reality score audits whether the answer actually generalizes. Weak verifiers produce high visible scores and large Goodhart gaps; the robust verifier reduces the gap by combining held-out checks, false-premise probes, freshness, citation quality, atomic support, usefulness, and overfit penalties.

The demo also translates the lesson to Answer Engine Optimization: optimizing mention count, citation count, or target-answer mimicry can Goodhart unless balanced by source-backed, held-out evaluation.

## GitHub repo

https://github.com/skyzer/self-improving-agent-verifier-optimizer

## Live demo

https://skyzer.github.io/self-improving-agent-verifier-optimizer/

## Additional photos

Overfit failure / large gap:

https://github.com/skyzer/self-improving-agent-verifier-optimizer/raw/main/media/01-overfit-goodhart-gap.png

Robust verifier / small gap:

https://github.com/skyzer/self-improving-agent-verifier-optimizer/raw/main/media/02-robust-small-gap.png

Hidden false-premise scenario:

https://github.com/skyzer/self-improving-agent-verifier-optimizer/raw/main/media/03-aeo-false-promise-robust.png

Earlier public screenshot:

https://github.com/skyzer/self-improving-agent-verifier-optimizer/raw/main/media/verifier-lab-main.png

The screenshots show the public GitHub Pages demo in multiple states: a weak verifier producing a high score but bad hidden reality score, the Robust verifier producing a small gap, and a hidden Answer Engine Optimization false-premise scenario.

## Suggested video/photo description

The screenshot/video should show the SIA two levers scenario, then switch between Bad, Easy, Overfit, and Robust verifier lenses. The weak lenses should show high verifier scores with low hidden reality scores, while Robust should show a small Goodhart gap.

## Tracks

Tracks: Research Track, Applied AI Track
