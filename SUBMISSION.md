# Oatmeal Submission

## Project title

Verifier Design Lab for SIA Agents

## One-line summary

An interactive Goodhart lab showing how self-improving agents can optimize weak verifier scores while failing hidden reality checks, and how robust verifier design mitigates the failure.

## Description

Verifier Design Lab is a static interactive demo for the Self Improving AI (SIA) Agents hackathon. It simulates a self-improving loop choosing candidate answers under four verifier designs: bad proxy, easy surface, overfit visible-set, and robust guardrail.

The same candidate answer can receive different visible-to-optimizer verifier scores depending on the active lens, while a hidden reality score audits whether the answer actually generalizes. Weak verifiers produce high visible scores and large Goodhart gaps; the robust verifier reduces the gap by combining held-out checks, false-premise probes, freshness, citation quality, atomic support, usefulness, and overfit penalties.

The demo also translates the lesson to Answer Engine Optimization: optimizing mention count, citation count, or target-answer mimicry can Goodhart unless balanced by source-backed, held-out evaluation.

## GitHub repo

https://github.com/skyzer/self-improving-agent-verifier-optimizer

## Live demo

https://skyzer.github.io/self-improving-agent-verifier-optimizer/

## Temporary fallback demo

https://as-mac-mini.cobbler-procyon.ts.net/verifier/

This is a tailnet-hosted fallback and should not be used as the only judge-facing URL.

## Additional photo

https://github.com/skyzer/self-improving-agent-verifier-optimizer/blob/main/media/verifier-lab-main.png

The screenshot shows the public GitHub Pages demo with the Robust verifier selected, including visible-to-optimizer verifier score, hidden reality score, and Goodhart gap labels.

## Suggested video/photo description

The screenshot/video should show the SIA two levers scenario, then switch between Bad, Easy, Overfit, and Robust verifier lenses. The weak lenses should show high verifier scores with low hidden reality scores, while Robust should show a small Goodhart gap.

## Tracks

Tracks: Research Track, Applied AI Track
