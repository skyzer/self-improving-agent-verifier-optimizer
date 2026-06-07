# Verifier Design Lab for Self-Improving Agents

Interactive Goodhart demo for the **Self Improving AI (SIA) Agents** hackathon.

- **Live demo:** https://skyzer.github.io/self-improving-agent-verifier-optimizer/
- **Repository:** https://github.com/skyzer/self-improving-agent-verifier-optimizer
- **Temporary local/tailnet demo:** https://as-mac-mini.cobbler-procyon.ts.net/verifier/
- **Submission tracks:** Research Track, Applied AI Track
- **Additional photo:** https://github.com/skyzer/self-improving-agent-verifier-optimizer/blob/main/media/verifier-lab-main.png

## What it shows

Self-improving agents optimize whatever verifier they are given. If the verifier is a weak proxy, an easy surface check, or overfit to visible examples, the optimizer can raise the visible verifier score while getting worse on hidden reality checks.

Verifier Design Lab makes that failure mode concrete:

- Choose a scenario.
- Compare candidate answer styles.
- Switch the active verifier lens: **Bad**, **Easy**, **Overfit**, or **Robust**.
- Watch the optimizer select the answer with the highest visible-to-optimizer verifier score.
- Compare that selected answer against the private **hidden reality score**.

The key lesson is the Goodhart warning from SIA: harness updates and weight updates are powerful, but both optimize the scoring function you give them.

## Official SIA resources

- Paper/deck from the SIA author: https://docs.google.com/presentation/d/1SORCjOiB52rWrYorW-O_8pVr4mpF0IM5u0j5bmqBeYY/edit?slide=id.g3e996067c3c_0_61&referrer=luma#slide=id.g3e996067c3c_0_61
- GitHub repo: https://github.com/hexo-ai/sia
- Repo README says SIA is a self-improving loop where a language-model agent updates both the harness and the weights of a task-specific agent.

## The verifier lenses

The lab compares four verifier designs:

- **Bad proxy verifier** — rewards answer length, keyword hits, and citation count.
- **Easy surface verifier** — rewards one approved phrase plus any citation.
- **Overfit visible-set verifier** — rewards memorized visible training examples.
- **Robust guardrail verifier** — balances atomic support, false-premise handling, freshness, citation quality, usefulness, and overfit penalties.

Each verifier chooses a different “optimized” answer style. Weak verifiers select brittle answers with high verifier scores and low hidden reality scores. The robust verifier selects the grounded answer and keeps the Goodhart gap small.

## Scenarios

The interactive bench includes five demo prompts:

1. **SIA two levers** — basic concept check.
2. **Answer Engine Optimization false promise** — refusal of an impossible/false premise.
3. **Citation-count trap** — citation quality beats raw citation count.
4. **Freshness drift** — launch-day checks go stale after product/source changes.
5. **Goodhart paraphrase** — concept understanding beats memorized slogans.

## How to read the scores

- **Verifier score / `{Lens} score`** — the visible-to-optimizer score produced by the currently selected verifier lens.
- **Hidden reality score** — a private held-out audit score that the optimizer does not see.
- **Optimizer** — the selector/chaser that picks the answer with the highest visible verifier score.
- **Goodhart gap** — verifier score minus hidden reality score. A large positive gap means the answer looks good to the verifier but fails the hidden audit.

## Answer Engine Optimization implication

For Answer Engine Optimization, do **not** optimize one shallow metric.

Bad verifier examples:

- mention count alone;
- citation count alone;
- exact target-answer phrase matching;
- visible prompt-suite pass rate only.

Better verifier stack:

- held-out prompt suite;
- atomic factual support;
- citation relevance/support;
- false-premise tests;
- freshness checks;
- competitor/fairness checks;
- human/client review of target claims.

## Run locally

### Python

```bash
python3 verifier_lab.py --validate
python3 -m http.server 8787 --bind 0.0.0.0
```

Then open:

```text
http://127.0.0.1:8787/index.html
```

### Docker / Compose

```bash
docker compose up --build -d
```

Then open:

```text
http://127.0.0.1:8787/index.html
```

Stop it with:

```bash
docker compose down
```

The container serves the static demo on `0.0.0.0:8787` and runs `python3 verifier_lab.py --validate` plus unit tests during image build. Compose bind-mounts this directory into `/app`, so regenerating `index.html` updates the running webapp without rebuilding the image.

## Run tests

```bash
python3 verifier_lab.py --validate
python3 -m unittest discover -s tests -v
```

## Demo talk track

1. Open the page and start from the left scenario rail.
2. Click through the scenarios: SIA two levers, Answer Engine Optimization false promise, citation-count trap, freshness drift, and Goodhart paraphrase.
3. Watch the middle column update: it always shows all candidate answers for the selected scenario, with the optimizer's chosen answer highlighted.
4. Use the verifier lens in the top right: **Bad**, **Easy**, **Overfit**, then **Robust**.
5. Show that weak lenses choose brittle policies with high verifier scores and low hidden reality scores.
6. Switch to **Robust** and show the grounded policy wins and the Goodhart gap nearly disappears.
7. Connect back to SIA: harness updates and weight updates are powerful, but both optimize the verifier.
8. Connect to Answer Engine Optimization: bad metrics like mention count, citation count, or target-answer mimicry can Goodhart unless balanced by held-out prompts and atomic support checks.

## Visual design

The browser demo uses the **Handhold Minimal** design system: black text on white surfaces, thin neutral borders, small radii, no shadows, no gradients, compact system typography, and red only for risky/failing states.

## Files

- `verifier_lab.py` — deterministic scoring engine + HTML/data generator.
- `index.html` — interactive browser demo served by GitHub Pages.
- `data/results.json` — generated score matrix.
- `tests/test_verifier_lab.py` — regression tests proving expected Goodhart behavior and UI wording.
- `ux-structure-options.html` — alternate UX structure ideas.
- `ux-fit-options.html` — compact layout alternatives.
- `Dockerfile` / `docker-compose.yml` — local containerized serving.
- `SUBMISSION.md` — Oatmeal submission copy.
- `media/verifier-lab-main.png` — public screenshot/additional photo for judging materials.

## License

MIT.
