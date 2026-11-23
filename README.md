# Fed Policy Agent

[中文说明](README_ZH.md)

How can a multi-agent LLM system forecast central-bank policy decisions while enforcing strict point-in-time information cutoffs and verifiable structured outputs?

This repository is the public MVP of a five-agent Federal Reserve forecasting and policy-simulation workflow. It combines structured JSON schemas, reusable policy-persona Skills, temporal-leakage gates, counterfactual scenario evaluation, and training utilities for Qwen LoRA SFT-to-GRPO post-training with LLM-as-Teacher reward scoring.

- `src/fed_game/workflow.py` and `agents.py`: five-agent orchestration and structured decision schemas (`schemas.py`).
- `src/fed_game/persona.py`, `skills.py`, and `policy_personas/warsh_policy/SKILL.md`: modular policy-persona Skill loading.
- `src/fed_game/evaluation.py` and `counterfactual.py`: backtest scoring and counterfactual policy runs.
- `src/fed_game/training/`: data split, reward, SFT, DAPT, and GRPO training modules (`splits.py`, `rewards.py`, `sft.py`, `grpo.py`).

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` executes the offline multi-agent workflow with deterministic mock responses, validating schema enforcement, persona Skill loading, reward scoring, and temporal split checks.

## Scope

This is the public MIT-licensed MVP runnable offline without API keys. Full training checkpoints and historical FOMC corpora are available upon request.
