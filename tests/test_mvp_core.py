"""Core contract tests for Fed Policy Agent MVP (stdlib unittest)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fed_game.persona import load_policy_persona  # noqa: E402
from fed_game.schemas import StrategyVector  # noqa: E402
from fed_game.training.rewards import score_completion_components  # noqa: E402
from fed_game.training.splits import split_jsonl_temporally  # noqa: E402
from models.fomc_labels import build_fomc_label_df  # noqa: E402


class FedPolicyAgentMVPTests(unittest.TestCase):
    def test_persona_prior_consistency(self) -> None:
        persona = load_policy_persona("policy_personas/warsh_policy/SKILL.md")
        self.assertEqual(persona.role_id, "usa_warsh")
        self.assertEqual(persona.consistency_score(StrategyVector.from_dict(persona.priors))["score"], 1.0)

    def test_temporal_split_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "rows.jsonl"
            rows = [
                {"messages": [{"role": "user", "content": json.dumps({"quarter": "2019Q4"})}], "metadata": {}},
                {"messages": [{"role": "user", "content": json.dumps({"quarter": "2020Q1"})}], "metadata": {}},
                {"messages": [{"role": "user", "content": json.dumps({"quarter": "2024Q2"})}], "metadata": {}},
            ]
            src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            summary = split_jsonl_temporally(src, tmp / "splits")
            self.assertEqual(summary["splits"]["train"]["rows"], 1)
            self.assertEqual(summary["splits"]["val"]["rows"], 1)
            self.assertEqual(summary["splits"]["test"]["rows"], 1)

    def test_reward_leakage_gate_and_missing_data_boundary(self) -> None:
        old = os.environ.pop("FOMC_LABELS_CSV", None)
        try:
            with self.assertRaises(FileNotFoundError):
                build_fomc_label_df()

            with tempfile.TemporaryDirectory() as td:
                csv_path = Path(td) / "synthetic.csv"
                csv_path.write_text("date,decision,ff_lower,ff_upper,desc\n2024-06-12,0,5.25,5.50,demo\n", encoding="utf-8")
                os.environ["FOMC_LABELS_CSV"] = str(csv_path)
                comp = {
                    "equilibrium_strategy": {
                        "hawkish_signal_prob": 0.7,
                        "rate_hike_25bp_prob": 0.4,
                        "hold_with_hawkish_statement_prob": 0.5,
                        "remove_forward_guidance_prob": 0.6,
                        "easing_signal_prob": 0.1,
                        "liquidity_support_prob": 0.2,
                        "trade_or_sanction_pressure_prob": 0.2,
                    },
                    "fed_prediction": {"hike_25bp": 0.55, "hold": 0.35, "cut_25bp": 0.10},
                    "converged": True,
                    "rounds": 6,
                    "evidence_chain": [{"published_at": "2024-06-12", "claim": "as-of"}],
                }
                ok = score_completion_components(json.dumps(comp), quarter="2024Q2", quarter_end="2024-06-30")
                self.assertEqual(ok["no_future_leakage"], 1.0)

                leaky = dict(comp)
                leaky["evidence_chain"] = [{"published_at": "2024-12-18", "claim": "future"}]
                bad = score_completion_components(json.dumps(leaky), quarter="2024Q2", quarter_end="2024-06-30")
                self.assertLess(bad["no_future_leakage"], ok["no_future_leakage"])
        finally:
            if old is None:
                os.environ.pop("FOMC_LABELS_CSV", None)
            else:
                os.environ["FOMC_LABELS_CSV"] = old


if __name__ == "__main__":
    unittest.main()
