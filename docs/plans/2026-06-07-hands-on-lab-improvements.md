# Hands-On Verifier Lab Improvements Implementation Plan

> **For Hermes:** Use test-driven-development skill to implement this plan task-by-task.

**Goal:** Turn the Verifier Design Lab from a strong explainer into a hands-on learning lab where viewers can trace a failure, compare weak vs robust verifiers, apply real-world scenarios, use a pre-training checklist, experiment with controls, and answer quiz prompts.

**Architecture:** Keep `verifier_lab.py` as the source of truth. Add deterministic data builders for trace/scenario/quiz/checklist content, render new HTML sections from the existing static generator, regenerate `index.html` and `data/results.json`, then update README/SUBMISSION/LEARNING docs.

**Tech Stack:** Dependency-free Python static generator, vanilla HTML/CSS/JS, Python `unittest` regression tests, GitHub Pages.

---

### Task 1: Add regression tests for new data and rendered UI

**Objective:** Prove the requested sections exist before implementing them.

**Files:**
- Modify: `tests/test_verifier_lab.py`

**Step 1: Write failing tests**

Add tests asserting:
- `evaluate()` returns `failure_traces`, `real_world_scenarios`, `builder_checklist`, `playground_scenarios`, and `quiz_questions`.
- Weak verifier traces recommend `H→W`, Robust permits `W`, and every trace has the five teaching steps.
- Generated HTML contains section headings and JS hooks:
  - `Watch one self-improvement loop go wrong`
  - `Bad verifier vs robust verifier`
  - `Real-world failure cards`
  - `Before you train weights, ask this`
  - `Mini playground`
  - `This is not anti-training`
  - `What should the agent do next?`

**Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_verifier_lab.VerifierLabTests -v`
Expected: FAIL because the new data keys and UI markers are missing.

---

### Task 2: Implement deterministic learning data builders

**Objective:** Add structured data for the trace, real-world cards, checklist, playground, and quiz.

**Files:**
- Modify: `verifier_lab.py`

**Step 1: Add builders**

Add pure functions:
- `build_failure_traces(winners, lever_decisions)`
- `build_real_world_scenarios()`
- `build_builder_checklist()`
- `build_playground_scenarios()`
- `build_quiz_questions()`

**Step 2: Wire into `evaluate()`**

Add the new keys to the returned JSON object.

**Step 3: Run tests**

Run: `python3 -m unittest tests.test_verifier_lab.VerifierLabTests -v`
Expected: data tests pass; HTML marker tests still fail until rendering is added.

---

### Task 3: Render the failure trace and weak-vs-robust comparison

**Objective:** Make the user see the self-improvement loop fail step by step.

**Files:**
- Modify: `verifier_lab.py`

**Step 1: Add CSS**

Add compact classes for trace steps, comparison controls, and side-by-side score panels.

**Step 2: Add HTML containers**

Add sections after `Which lever should the agent pull?`:
- `failureTraceSection` / `failureTracePanel`
- `comparisonSection` / `comparisonPanel`

**Step 3: Add JS renderers**

Add:
- `renderFailureTrace()` using the active verifier.
- `renderComparison()` with a `comparisonVerifier` state toggling `bad_proxy` vs `robust_guardrail`.

**Step 4: Call from `render()`**

Call both renderers after `renderLeverSection()`.

---

### Task 4: Render scenario cards, checklist, playground, and quiz

**Objective:** Convert the concept into practical learning artifacts.

**Files:**
- Modify: `verifier_lab.py`

**Step 1: Add static sections**

Render:
- real-world scenario cards for code agent, RAG agent, browser agent, and LLM fine-tune;
- builder checklist;
- anti-training explanation;
- inspiration note.

**Step 2: Add playground controls**

Add scenario select and sliders:
- verifier strictness;
- hidden audit strength;
- optimizer pressure.

JS computes visible score, hidden score, gap, recommended lever, and blind `W` regret.

**Step 3: Add quiz cards**

Add multiple-choice buttons that reveal the correct lever and rationale.

---

### Task 5: Regenerate and verify locally

**Objective:** Make generated output match source and tests.

**Files:**
- Generated: `index.html`
- Generated: `data/results.json`

**Commands:**

```bash
python3 verifier_lab.py --validate
python3 -m unittest discover -s tests -v
```

Expected: PASS.

---

### Task 6: Update docs

**Objective:** Keep README/SUBMISSION/LEARNING aligned with the new demo.

**Files:**
- Modify: `README.md`
- Modify: `SUBMISSION.md`
- Modify: `LEARNING.md`

**Details:**
- Add the new hands-on learning sections to the repo description.
- Update demo talk track.
- Keep public docs free of private networking links.

---

### Task 7: Browser/public verification, commit, push

**Objective:** Ship only after verified build and UI.

**Commands:**

```bash
python3 verifier_lab.py --validate
python3 -m unittest discover -s tests -v
git diff --check
git status --short
git add verifier_lab.py index.html data/results.json tests/test_verifier_lab.py README.md SUBMISSION.md LEARNING.md docs/plans/2026-06-07-hands-on-lab-improvements.md
git commit -m "feat: add hands-on verifier lab sections"
git push
```

Then wait for CI and GitHub Pages deployment, open the cache-busted Pages URL, click the new controls, check console errors, and visually verify the new sections are readable.
