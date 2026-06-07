# Learning Notes: Borrowed SIA-Lever Ideas

This repo intentionally borrows **ideas and framing** from `mdkrasnow/sia-lever`, not source code. That repo did not expose a license file when inspected, so the safe path is to restate the concepts in original demo code and copy.

## What we borrowed conceptually

1. **Lever attribution**
   - A self-improving agent should not only ask “which answer scores highest?”
   - It should ask “which intervention is safe now?”
   - Choices:
     - `H` — fix harness/verifier/scaffold.
     - `W` — train model weights against the current verifier.
     - `H→W` — fix verifier first, then train.

2. **Do not train weights against a bad verifier**
   - If the verifier rewards a shortcut, a weight update can entrench the shortcut.
   - In the demo, Bad/Easy/Overfit lenses therefore recommend `H→W`.

3. **Oracle sandwich check**
   - Ask whether a known-good answer/model passes the current harness.
   - If a known-good answer fails or is under-ranked, the harness needs repair before weight training.

4. **Shortcut signal**
   - Look for cases where the visible score rises while hidden reality falls.
   - This is the UI version of detecting a model that “solves” the visible task by exploiting the wrong signal.

5. **Honest claim discipline**
   - This demo is a deterministic teaching lab, not a measured training benchmark.
   - The `regret_if_w` values are toy score deltas to teach the decision, not empirical model-training outcomes.

## How to explain the upgraded demo

1. Start with the Scenario-first lab bench.
2. Show that Bad/Easy/Overfit verifiers select brittle answers.
3. Use the verifier-loop graph to show visible score rising while hidden reality falls.
4. Scroll to **Which lever should the agent pull?**.
5. Explain:
   - weak verifier → choose `H→W`
   - robust verifier → `W` is allowed, but hidden audits still stay private
6. Use **Watch one self-improvement loop go wrong** to narrate the exact sequence:
   - candidate answer
   - visible verifier score
   - optimizer selection
   - hidden audit
   - lever attribution
7. Use **Bad verifier vs robust verifier** for the quick before/after toggle.
8. Use the real-world failure cards to transfer the lesson to code agents, RAG agents, browser agents, and LLM fine-tunes.
9. Use the builder checklist before proposing any weight update.
10. Use the playground sliders to show why high optimizer pressure plus weak verification creates blind `W` regret.
11. Use the quiz cards to make the viewer choose the next lever.
12. Connect to SIA:
   - harness updates define what weight updates optimize
   - bad harness first means bad weight training later
13. Connect to Answer Engine Optimization:
   - if the metric is shallow, optimizing it creates fake progress
   - fix the evaluator before optimizing the model/process

## Next improvements to consider

- Capture new screenshots for the failure trace / playground / quiz sections.
- Add a tiny downloadable toy trace JSON for users who want to write their own lever classifier.
- Add public/private task JSON examples to make the demo more SIA-native.
- Add a small “design your own verifier” form that composes checklist items into a scoring rule.
- If we later use any actual SIA-Lever code, first verify its license or get permission.
