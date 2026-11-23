"""Minimal runnable synthetic demo for Fed Policy Agent MVP.

Exercises the real persona loader, structured best-response / equilibrium
contracts (with a deterministic offline teacher stub), temporal split boundary,
GRPO reward + temporal-leakage gate, and explicit missing-data boundary.

NOTE: Uses in-memory synthetic data only. This is a software/contract
demonstration, NOT a historical FOMC forecasting backtest.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fed_game.clusters import get_role  # noqa: E402
from fed_game.persona import load_policy_persona  # noqa: E402
from fed_game.schemas import AgentMemory, StrategyVector  # noqa: E402
from fed_game.skills import LLMBestResponseSkill, LLMEquilibriumJudge  # noqa: E402
from fed_game.training.rewards import score_completion_components  # noqa: E402
from fed_game.training.splits import split_jsonl_temporally  # noqa: E402
from models.fomc_labels import build_fomc_label_df  # noqa: E402


class DeterministicDemoTeacher:
    """Offline stub returning schema-compliant JSON for demonstration only."""

    def chat_json(self, messages, schema_hint=None):
        schema_type = (schema_hint or {}).get("type")
        if schema_type == "policy_best_response":
            return {
                "strategy": {
                    "hawkish_signal_prob": 0.66,
                    "rate_hike_25bp_prob": 0.31,
                    "hold_with_hawkish_statement_prob": 0.72,
                    "remove_forward_guidance_prob": 0.61,
                    "easing_signal_prob": 0.08,
                    "liquidity_support_prob": 0.16,
                    "trade_or_sanction_pressure_prob": 0.23,
                },
                "rationale": "Synthetic demo: inflation credibility dominates with optionality.",
            }
        if schema_type == "policy_payoff_judgement":
            return {
                "payoff": 0.74,
                "regret_estimate": 0.01,
                "best_unilateral_deviation": {
                    "strategy_change": "slightly lower hike probability",
                    "alternative_payoff": 0.75,
                },
                "reasoning": "Synthetic demo: near-optimal response with low regret.",
                "confidence": 0.80,
            }
        if schema_type == "nash_equilibrium_check":
            return {
                "is_nash_equilibrium": "true",
                "max_profitable_deviation_gain": 0.01,
                "profitable_deviations": [],
                "reasoning": "Synthetic demo: no profitable unilateral deviation.",
                "confidence": 0.77,
            }
        raise ValueError(f"Unexpected schema_type: {schema_type}")


def main() -> None:
    print("=== Fed Policy Agent MVP (synthetic demo) ===")

    # 1. Policy persona & consistency
    persona = load_policy_persona("policy_personas/warsh_policy/SKILL.md")
    consistency = persona.consistency_score(StrategyVector.from_dict(persona.priors))
    print(
        f"[1/5] Persona loaded: {persona.persona_id} (role={persona.role_id}, "
        f"prior_consistency={consistency['score']:.3f})"
    )

    # 2. Multi-agent best-response & Nash equilibrium contract (offline stub)
    teacher = DeterministicDemoTeacher()
    br_skill = LLMBestResponseSkill(teacher, use_llm_payoff=True)
    eq_judge = LLMEquilibriumJudge(teacher)
    role = get_role("usa_warsh")
    context = {
        "quarter": "2024Q2",
        "var_vecm_briefing": {
            "inflation_cpi_yoy": 3.4,
            "unemployment": 4.0,
            "taylor_gap_pp": 0.8,
            "ect_signal": "near_equilibrium",
        },
        "policy_persona": persona.to_prompt_payload(),
    }
    proposal = br_skill.run(
        role=role,
        memory=AgentMemory(role_id=role.role_id, cluster_id="USA"),
        context=context,
        opponent_strategies={"CHN": StrategyVector(trade_or_sanction_pressure_prob=0.4)},
        evidence=[{"title": "Synthetic FOMC note", "text": "Inflation remains elevated."}],
        round_id=1,
    )
    eq = eq_judge.check(
        quarter="2024Q2",
        round_id=10,
        cluster_strategies={"USA": proposal.strategy, "CHN": StrategyVector()},
        proposals=[proposal],
        var_vecm_briefing=context["var_vecm_briefing"],
        heuristic={
            "max_strategy_distance": 0.01,
            "strategy_epsilon": 0.025,
            "max_deviation_gain": 0.01,
            "deviation_gain_tau": 0.02,
            "stable_rounds": 10,
            "stable_rounds_required": 10,
            "heuristic_passed": True,
        },
        role_personas={"usa_warsh": persona.to_prompt_payload(max_markdown_chars=1200)},
    )
    print(
        f"[2/5] Best-response & equilibrium contract: hawkish={proposal.strategy.hawkish_signal_prob:.2f}, "
        f"payoff={proposal.payoff_estimate:.2f}, regret={proposal.regret_estimate:.2f}, "
        f"is_nash={eq['is_nash_equilibrium']}"
    )

    # 3. Strict chronological split contract
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src = tmp / "synthetic_rows.jsonl"
        rows = [
            {"messages": [{"role": "user", "content": json.dumps({"quarter": "2019Q4"})}], "metadata": {}},
            {"messages": [{"role": "user", "content": json.dumps({"quarter": "2020Q1"})}], "metadata": {}},
            {"messages": [{"role": "user", "content": json.dumps({"quarter": "2024Q2"})}], "metadata": {}},
            {"messages": [{"role": "user", "content": "unparsed"}], "metadata": {}},
        ]
        src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        split_summary = split_jsonl_temporally(src, tmp / "splits")
        counts = {k: v["rows"] for k, v in split_summary["splits"].items()}
        print(
            f"[3/5] Temporal split contract: train={counts['train']}, "
            f"val={counts['val']}, test={counts['test']}, "
            f"unassigned={split_summary['unassigned']['rows']}"
        )

        # 4. GRPO reward & temporal-leakage gate on synthetic label CSV
        csv_path = tmp / "synthetic_fomc.csv"
        csv_path.write_text(
            "date,decision,ff_lower,ff_upper,desc\n"
            "2024-06-12,0,5.25,5.50,Synthetic demo label\n",
            encoding="utf-8",
        )
        old_env = os.environ.get("FOMC_LABELS_CSV")
        os.environ["FOMC_LABELS_CSV"] = str(csv_path)
        try:
            base = {
                "equilibrium_strategy": proposal.strategy.to_dict(),
                "fed_prediction": {"hike_25bp": 0.55, "hold": 0.35, "cut_25bp": 0.10},
                "converged": True,
                "rounds": 6,
                "evidence_chain": [
                    {
                        "source_id": "fed_fomc",
                        "url": "https://www.federalreserve.gov/example",
                        "published_at": "2024-06-12",
                        "claim": "Synthetic as-of evidence.",
                    }
                ],
            }
            leaky = dict(base)
            leaky["evidence_chain"] = [
                {
                    "source_id": "fed_fomc",
                    "published_at": "2024-12-18",
                    "claim": "Post-cutoff evidence (leakage test).",
                }
            ]
            collapsed = dict(base)
            collapsed["fed_prediction"] = {"hike_25bp": 0.0, "hold": 1.0, "cut_25bp": 0.0}

            s_ok = score_completion_components(
                json.dumps(base), quarter="2024Q2", role_id="usa_warsh", current_round=6, quarter_end="2024-06-30"
            )
            s_leak = score_completion_components(
                json.dumps(leaky), quarter="2024Q2", role_id="usa_warsh", current_round=6, quarter_end="2024-06-30"
            )
            s_col = score_completion_components(
                json.dumps(collapsed), quarter="2024Q2", role_id="usa_warsh", current_round=6, quarter_end="2024-06-30"
            )
            print("[4/5] GRPO reward & temporal-leakage gate (synthetic):")
            print(
                f"      Valid as-of   : schema={s_ok['schema_valid']:.2f} | "
                f"no_leak={s_ok['no_future_leakage']:.2f} | "
                f"anti_hold={s_ok['not_always_hold']:.2f} | total={s_ok['weighted_total']:.3f}"
            )
            print(
                f"      Future leakage: schema={s_leak['schema_valid']:.2f} | "
                f"no_leak={s_leak['no_future_leakage']:.2f} | "
                f"anti_hold={s_leak['not_always_hold']:.2f} | total={s_leak['weighted_total']:.3f}"
            )
            print(
                f"      Always-hold   : schema={s_col['schema_valid']:.2f} | "
                f"no_leak={s_col['no_future_leakage']:.2f} | "
                f"anti_hold={s_col['not_always_hold']:.2f} | total={s_col['weighted_total']:.3f}"
            )
        finally:
            if old_env is None:
                os.environ.pop("FOMC_LABELS_CSV", None)
            else:
                os.environ["FOMC_LABELS_CSV"] = old_env

    # 5. Missing-data boundary
    old_env = os.environ.pop("FOMC_LABELS_CSV", None)
    try:
        try:
            build_fomc_label_df()
            raise AssertionError("Expected FileNotFoundError when FOMC_LABELS_CSV is unset")
        except FileNotFoundError:
            print("[5/5] Missing-data boundary verified: FileNotFoundError raised without FOMC_LABELS_CSV.")
    finally:
        if old_env is not None:
            os.environ["FOMC_LABELS_CSV"] = old_env


if __name__ == "__main__":
    main()
