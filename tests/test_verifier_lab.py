import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import verifier_lab


class VerifierLabTests(unittest.TestCase):
    def setUp(self):
        self.data = verifier_lab.evaluate()

    def test_expected_goodhart_winners(self):
        verifier_lab.validate_expected_behavior(self.data)

    def test_weak_verifiers_have_large_hidden_gaps(self):
        winners = self.data["winners"]
        for verifier_id in ["bad_proxy", "easy_surface", "overfit_visible"]:
            with self.subTest(verifier=verifier_id):
                self.assertGreaterEqual(winners[verifier_id]["goodhart_gap"], 35)
                self.assertLess(winners[verifier_id]["hidden_reality_score"], 25)

    def test_robust_verifier_tracks_hidden_reality(self):
        winner = self.data["winners"]["robust_guardrail"]
        self.assertEqual(winner["policy_id"], "robust_grounded_agent")
        self.assertLessEqual(abs(winner["visible_score"] - winner["hidden_reality_score"]), 8)
        self.assertGreater(winner["hidden_reality_score"], 80)

    def test_generated_html_and_json_are_written(self):
        out = ROOT / "build-test"
        paths = verifier_lab.write_outputs(out)
        index = pathlib.Path(paths["index"])
        results = pathlib.Path(paths["results"])
        self.assertTrue(index.exists())
        self.assertTrue(results.exists())
        html = index.read_text(encoding="utf-8")
        self.assertIn("Verifier Design Lab", html)
        self.assertIn("LAB_RESULTS", html)
        self.assertIn("Scenario-first lab bench", html)
        self.assertIn("lab-grid", html)
        self.assertIn("class=\"term-help\">Goodhart", html)
        self.assertIn("Goodhart: when a metric becomes the target", html)
        self.assertIn("data-tooltip=\"Goodhart:", html)
        self.assertNotIn("title=\"Goodhart:", html)
        self.assertIn("resultPanel", html)
        self.assertIn("overflow: auto", html)
        self.assertIn("overscroll-behavior: contain", html)
        self.assertIn(".winner-answer { padding: 0; border-radius: 0; background: transparent; border: 0; }", html)
        self.assertNotIn("Result under", html)
        self.assertIn("Candidate answers for this scenario", html)
        self.assertIn("Basic concept check", html)
        self.assertIn("False-claim refusal", html)
        self.assertIn("Citation quality check", html)
        self.assertIn("Freshness drift check", html)
        self.assertIn("Paraphrase understanding check", html)
        self.assertIn("Visible to optimizer", html)
        self.assertIn("Hidden held-out test", html)
        self.assertIn("The selected verifier produces the visible-to-optimizer score", html)
        self.assertIn("candidate rows are rescored by that active lens", html)
        self.assertIn("Scoreboard: active verifier score vs hidden reality score", html)
        self.assertIn("Verifier scores update when you switch lenses", html)
        self.assertIn("Verifier score", html)
        self.assertIn("Hidden reality score", html)
        self.assertIn("visible-to-optimizer verifier score", html)
        self.assertIn("${escapeHtml(verifier.label)} score ${fmt(visible)}", html)
        self.assertIn("hidden reality score ${fmt(hidden)}", html)
        self.assertIn("${escapeHtml(verifier.label)} score</label>", html)
        self.assertNotIn("visible ${fmt(visible)}", html)
        self.assertIn("How to read the verifier loop", html)
        self.assertIn("verifier/evaluation loop", html)
        self.assertIn("independent held-out score", html)
        self.assertIn("Why Robust is not magic", html)
        self.assertIn("In this toy demo, Robust is hardcoded", html)
        self.assertIn("Hidden reality is a private held-out audit", html)
        self.assertIn("hidden reality score stays fixed", html)
        self.assertIn("An independent held-out score the optimizer does not see", html)
        self.assertNotIn("${escapeHtml(e.kind)} · ${escapeHtml(e.split)}", html)
        self.assertIn("lens-row", html)
        self.assertIn('data-formula=\"${escapeHtml(v.formula)}\"', html)
        self.assertNotIn('title=\"${escapeHtml(v.name)}', html)
        self.assertIn('"formula": "score = 12 + 0.22×words', html)
        self.assertIn('"formula": "score = 15 + 48×exact_phrase', html)
        self.assertIn("score = 8 + 72×memorized_training_answer", html)
        self.assertNotIn("memorized_visible_answer", html)
        self.assertIn("score = 100×(0.45×visible_quality", html)
        self.assertNotIn("formula-block", html)
        self.assertNotIn("<span>Rule:</span>", html)
        self.assertNotIn("Interactive demo", html)
        self.assertNotIn("Pick a verifier above", html)
        self.assertIn("Bad answer", html)
        self.assertIn("Easy answer", html)
        self.assertIn("Overfit answer", html)
        self.assertIn("Robust answer", html)
        for old_label in ("Keyword/citation stuffer", "Stuffer", "Approved phrase bot", "Phrase bot", "Memorized training bot", "Memorizer", "Robust grounded agent"):
            self.assertNotIn(old_label, html)
        self.assertIn("examplePanel", html)
        self.assertIn('rel="icon" href="./favicon.svg"', html)
        self.assertIn('rel="apple-touch-icon" href="./apple-touch-icon.png"', html)
        self.assertIn('class="topic-line"', html)
        self.assertNotIn("SIA " + "Hackathon Demo", html)
        self.assertNotIn("pill" + "-row", html)
        self.assertNotIn('class="' + "pill" + '"', html)

    def test_favicon_assets_exist(self):
        for name in ["favicon.svg", "favicon.ico", "favicon-32x32.png", "apple-touch-icon.png", "favicon-512.png"]:
            with self.subTest(asset=name):
                path = ROOT / name
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 100)

    def test_policy_marker_colors_are_neutral(self):
        colors = {policy["color"] for policy in self.data["policies"]}
        self.assertEqual(colors, {"#000000"})

    def test_ux_structure_options_page_exists(self):
        path = ROOT / "ux-structure-options.html"
        self.assertTrue(path.exists())
        html = path.read_text(encoding="utf-8")
        self.assertIn("Verifier Design Lab navigation alternatives", html)
        self.assertIn("Option A", html)
        self.assertIn("Option B", html)
        self.assertIn("Option C", html)
        self.assertIn("Option D", html)
        self.assertIn("Compare all verifiers for one prompt", html)
        self.assertIn("Handhold Minimal", html)

    def test_ux_fit_options_page_exists(self):
        path = ROOT / "ux-fit-options.html"
        self.assertTrue(path.exists())
        html = path.read_text(encoding="utf-8")
        self.assertIn("MacBook-fit options after Option A", html)
        self.assertIn("Three-pane lab bench", html)
        self.assertIn("Answer-first split pane", html)
        self.assertIn("Verifier comparison board", html)
        self.assertIn("Presentation mode with progressive reveal", html)
        self.assertIn("candidate answers", html)
        self.assertIn("Handhold Minimal", html)

    def test_no_download_zip_artifact_is_generated(self):
        self.assertFalse((ROOT / "verifier-design-lab-demo.zip").exists())

    def test_compose_bind_mounts_app_for_static_reload(self):
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("./:/app:ro", compose)

    def test_handhold_minimal_design_applied(self):
        out = ROOT / "build-test"
        paths = verifier_lab.write_outputs(out)
        html = pathlib.Path(paths["index"]).read_text(encoding="utf-8")
        self.assertIn('data-design-system="Handhold Minimal"', html)
        self.assertIn("color-scheme: light", html)
        self.assertIn("--color-primary: #000000", html)
        self.assertIn("--color-neutral: #E5E7EB", html)
        self.assertIn("--radius-sm: 4px", html)
        for forbidden in ("radial" + "-gradient", "linear" + "-gradient", "box" + "-shadow"):
            self.assertNotIn(forbidden, html)

    def test_demo_examples_cover_every_policy(self):
        examples = self.data["demo_examples"]
        responses = self.data["example_responses"]
        policy_ids = {policy["id"] for policy in self.data["policies"]}
        self.assertGreaterEqual(len(examples), 5)
        for example in examples:
            with self.subTest(example=example["id"]):
                self.assertIn(example["id"], responses)
                self.assertEqual(set(responses[example["id"]]), policy_ids)
                for answer in responses[example["id"]].values():
                    self.assertGreater(len(answer), 25)

    def test_examples_include_hidden_failure_modes(self):
        kinds = {example["kind"] for example in self.data["demo_examples"]}
        self.assertIn("false premise", kinds)
        self.assertIn("citation support", kinds)
        self.assertIn("drift", kinds)
        self.assertIn("paraphrase", kinds)

    def test_loop_proof_visible_only_goodharts_under_bad_proxy(self):
        proof = self.data["loop_proof"]["bad_proxy"]
        self.assertGreaterEqual(proof["loop_count"], 10)
        visible_only = proof["summary"]["visible_only"]
        gated = proof["summary"]["independent_gated"]
        self.assertGreaterEqual(visible_only["final_goodhart_gap"], 45)
        self.assertLessEqual(gated["final_goodhart_gap"], 10)
        self.assertGreater(gated["final_hidden_reality_score"], visible_only["final_hidden_reality_score"] + 40)
        self.assertGreaterEqual(gated["rejected_gaming_mutations"], 5)
        self.assertFalse(proof["summary"]["hidden_score_used_for_selection"])

    def test_loop_proof_keeps_independent_score_separate_from_hidden_reality(self):
        proof = self.data["loop_proof"]["bad_proxy"]
        first_step = proof["steps"][0]
        self.assertIn("independent_score", first_step["candidates"][0])
        self.assertIn("hidden_reality_score", first_step["candidates"][0])
        self.assertTrue(
            any(c["independent_score"] != c["hidden_reality_score"] for c in first_step["candidates"]),
            "independent verifier score should not be the same object as hidden reality",
        )

    def test_loop_proof_same_candidates_are_seen_by_both_arms(self):
        proof = self.data["loop_proof"]["overfit_visible"]
        for step in proof["steps"]:
            with self.subTest(loop=step["loop"]):
                self.assertEqual(step["visible_only"]["seen_candidate_ids"], step["candidate_ids"])
                self.assertEqual(step["independent_gated"]["seen_candidate_ids"], step["candidate_ids"])

    def test_lever_decisions_map_verifier_failures_to_actions(self):
        decisions = self.data["lever_decisions"]
        self.assertEqual(set(decisions), {"bad_proxy", "easy_surface", "overfit_visible", "robust_guardrail"})
        for verifier_id in ["bad_proxy", "easy_surface", "overfit_visible"]:
            with self.subTest(verifier=verifier_id):
                decision = decisions[verifier_id]
                self.assertEqual(decision["recommended_action"], "H_THEN_W")
                self.assertGreater(decision["regret_if_w"], 30)
                self.assertIn("oracle_sandwich", decision)
                self.assertIn("shortcut_signal", decision)
                self.assertIn("Do not train weights against a bad verifier", decision["reason"])
        robust = decisions["robust_guardrail"]
        self.assertEqual(robust["recommended_action"], "W")
        self.assertLessEqual(robust["regret_if_w"], 5)
        self.assertTrue(robust["oracle_sandwich"]["known_good_passes"])

    def test_loop_proof_ui_is_rendered(self):
        out = ROOT / "build-test"
        paths = verifier_lab.write_outputs(out)
        html = pathlib.Path(paths["index"]).read_text(encoding="utf-8")
        self.assertIn("Verifier loop proof", html)
        self.assertIn("Visible-only optimizer", html)
        self.assertIn("Independent-verifier-gated optimizer", html)
        self.assertIn("hidden reality lift", html)
        self.assertIn("proofChart", html)
        self.assertIn("loopTrace", html)
        self.assertIn("Rejected by independent verifier", html)
        self.assertIn("Independent gate score", html)
        self.assertIn("hidden reality audit", html)
        self.assertIn("independent gate ${fmt(step.independent_gated.independent_score)}", html)
        self.assertIn("normalized score: 0–100", html)
        self.assertIn("red: visible-only ${activeScoreLabel} score", html)
        self.assertIn("red dashed = visible-only optimizer — hidden reality audit", html)
        self.assertIn("black = independent-gated optimizer — ${escapeHtml(verifier.label)} score", html)
        self.assertIn("The independent gate score is not drawn as its own line", html)
        self.assertIn("0–100 scale:", html)
        self.assertIn("leverSection", html)
        self.assertIn("Which lever should the agent pull?", html)
        self.assertIn("H→W", html)
        self.assertIn("oracle sandwich", html)
        self.assertIn("Do not train weights against a bad verifier", html)


if __name__ == "__main__":
    unittest.main()
