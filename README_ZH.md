# Fed Policy Agent（央行政策多智能体系统）

[English](README.md)

多智能体大模型系统如何在严格执行时点截断（防时间泄漏）和结构化输出校验的前提下，完成央行货币政策预测与情景仿真？

本仓库是五类功能 Agent 协同的 FOMC 政策预测与仿真系统的公开 MVP。代码整合了结构化 JSON 协议、可复用政策人物 Skill、时间泄漏门控、反事实情景评估，以及面向 Qwen LoRA 的 SFT 到 GRPO 后训练与 LLM-as-Teacher 奖励打分模块。

- `src/fed_game/workflow.py` 与 `agents.py`：五智能体编排与结构化输出协议（`schemas.py`）。
- `src/fed_game/persona.py`、`skills.py` 与 `policy_personas/warsh_policy/SKILL.md`：模块化政策人物 Skill 加载机制。
- `src/fed_game/evaluation.py` 与 `counterfactual.py`：评测打分与反事实政策推演。
- `src/fed_game/training/`：时间切分、奖励函数、SFT、DAPT 与 GRPO 后训练模块（`splits.py`、`rewards.py`、`sft.py`、`grpo.py`）。

## 快速开始

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` 在离线模式下使用确定性模拟响应跑通完整多智能体流程，校验 JSON 结构、Skill 加载、奖励打分与时间切分门控。

## 范围说明

本仓库为无需 API 密钥即可离线运行的 MIT 公开 MVP，完整训练权重与历史 FOMC 语料按需提供。
