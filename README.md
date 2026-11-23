# Fed Policy Agent

Multi-agent monetary-policy forecasting prototype with typed strategy schemas, policy personas, chronological splits, and GRPO reward checks (MIT snapshot `91cfa92`).

Central-bank forecasting with LLMs commonly breaks on post-cutoff evidence leakage, malformed JSON outputs, and collapse onto predicting "hold" every quarter. This repository keeps the core modules built to catch those failures:

- `src/fed_game/persona.py` and `policy_personas/warsh_policy/SKILL.md`: loads structured policy priors and scores strategy consistency.
- `src/fed_game/skills.py` and `src/fed_game/self_play.py`: typed best-response, payoff judging, and Nash-equilibrium contracts.
- `src/fed_game/training/splits.py`: chronological train/val/test JSONL splitter.
- `src/fed_game/training/rewards.py`: GRPO reward scoring with `no_future_leakage`, evidence traceability, and `not_always_hold` penalties.
- `src/models/fomc_labels.py`: external CSV loader interface (no historical labels bundled).

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` runs on synthetic inputs with an offline stub teacher. It checks persona prior consistency (`1.000`), equilibrium schema validation, chronological split counts (`train=1, val=1, test=1, unassigned=1`), reward drops under post-cutoff leakage (`1.00` to `-0.25`) and always-hold collapse (`0.65` to `0.20`), and the missing-CSV error boundary.

## Scope

Live five-agent self-play and SFT/GRPO training require your own `DEEPSEEK_API_KEY`, an external FOMC CSV (`FOMC_LABELS_CSV`), and GPU training dependencies. MIT License.
