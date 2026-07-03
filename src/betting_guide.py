"""Fan-oriented betting guidance from model outputs (no market prices)."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.match_analyzer import MatchPrediction
from src.model import FootballOutcomeModel


@dataclass
class BettingGuide:
    """Conservative wagering suggestions for entertainment / small-stake use only."""

    primary_outcome: str
    primary_probability: float
    confidence_tier: str
    recommended_scoreline: str
    recommended_total_goals: float
    over_25_pick: str
    over_25_probability: float
    btts_pick: str
    btts_probability: float
    suggested_stake_pct: float
    suggested_stake_usd: float
    action_summary: str
    rationale: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)


_TIER_ZH = {"high": "高", "medium": "中", "low": "低"}


def _confidence_tier(probability: float, model_confidence: float) -> str:
    if probability >= 0.55 and model_confidence >= 0.58:
        return "high"
    if probability >= 0.45 and model_confidence >= 0.52:
        return "medium"
    return "low"


def _suggested_stake(probability: float, model_confidence: float, bankroll: float) -> tuple[float, float]:
    edge_proxy = max(0.0, probability - (1.0 / 3.0))
    kelly = edge_proxy * model_confidence * 0.25
    stake_pct = min(0.02, max(0.0, kelly))
    if probability < 0.42 or model_confidence < 0.52:
        stake_pct = 0.0
    return stake_pct, round(bankroll * stake_pct, 2)


def build_betting_guide(
    prediction: MatchPrediction,
    *,
    model: FootballOutcomeModel | None = None,
    bankroll: float = 70.0,
    min_probability: float = 0.40,
) -> BettingGuide:
    outcomes = [
        (prediction.team_a, prediction.prob_team_a_win),
        ("平局", prediction.prob_draw),
        (prediction.team_b, prediction.prob_team_b_win),
    ]
    outcomes.sort(key=lambda x: x[1], reverse=True)
    primary_name, primary_prob = outcomes[0]
    second_prob = outcomes[1][1]
    margin = primary_prob - second_prob

    tier = _confidence_tier(primary_prob, prediction.confidence)
    tier_zh = _TIER_ZH[tier]
    stake_pct, stake_usd = _suggested_stake(primary_prob, prediction.confidence, bankroll)

    sp = prediction.score_prediction
    if sp:
        recommended_score = sp.most_likely.label
        recommended_total = sp.expected_goals_a + sp.expected_goals_b
        over_pick = "大 2.5" if sp.over_25_prob >= 0.5 else "小 2.5"
        over_prob = sp.over_25_prob if sp.over_25_prob >= 0.5 else sp.under_25_prob
        btts_pick = "双方进球" if sp.btts_prob >= 0.5 else "单方零封"
        btts_prob = sp.btts_prob if sp.btts_prob >= 0.5 else 1.0 - sp.btts_prob
    else:
        recommended_score = "暂无"
        recommended_total = 0.0
        over_pick = over_prob = btts_pick = btts_prob = "暂无"

    rationale = [
        f"集成模型（Dixon–Coles + 有序 Logit + 多项 Logit）最看好 **{primary_name}**，概率 {primary_prob:.1%}。",
        f"与次选项的分差：{margin:.1%}。",
        f"Dixon–Coles 比分层：最热门精确比分 **{recommended_score}**（单一比分概率通常仅 ~10%），"
        f"预期总进球 **{recommended_total:.2f}**，"
        f"大 2.5 概率 **{sp.over_25_prob:.1%}**（所有≥3球比分之和）。",
    ]
    if prediction.component_probs:
        comp = prediction.component_probs
        w = comp.weights
        rationale.append(
            f"子模型权重 — Dixon–Coles {w[0]:.0%}，有序 Logit {w[1]:.0%}，多项 Logit {w[2]:.0%}。"
        )
    if prediction.uncertainty:
        u = prediction.uncertainty
        rationale.append(
            f"不确定性区间（约 90% 半宽）：{prediction.team_a} ±{u[0]:.1%}，"
            f"平局 ±{u[1]:.1%}，{prediction.team_b} ±{u[2]:.1%}。"
        )
    if model and model.bundle:
        m = model.bundle.metrics
        rationale.append(
            f"多项 Logit 历史回测 — 对数损失 {m.log_loss:.3f}，Brier {m.brier_score:.3f}，"
            f"准确率 {m.accuracy:.1%}（仅历史数据，不保证未来）。"
        )

    cautions = [
        "仅供球迷研究与娱乐，不构成投资建议。",
        "请勿投入无法承受损失的资金。",
        "下注前务必对比真实赔率；模型概率 ≠ 市场优势。",
    ]
    if tier == "low":
        cautions.append("置信度偏低 — 不建议下注，观望即可。")
        stake_pct, stake_usd = 0.0, 0.0
    if prediction.uncertainty and max(prediction.uncertainty) > 0.12:
        cautions.append("不确定性区间较宽 — 仅作软倾向参考。")

    if primary_prob < min_probability:
        action = "三项概率过于接近，暂无明确倾向。"
        primary_name = "暂无明确倾向"
    elif tier == "low":
        action = f"轻微倾向 **{primary_name}**（{primary_prob:.1%}），但置信度低 — 建议纸上谈兵。"
    else:
        action = (
            f"倾向 **{primary_name}**（{primary_prob:.1%}，{tier_zh}置信度）。"
            f" 比分倾向：**{recommended_score}**。"
            f" 进球：**{over_pick}**（{over_prob:.1%}）。"
            f" BTTS：**{btts_pick}**（{btts_prob:.1%}）。"
        )
        if stake_usd > 0:
            action += f" 娱乐型仓位上限：**${stake_usd:.2f}**（约资金的 {stake_pct:.1%}）。"
        else:
            action += " 建议仓位：**$0**（本场观望）。"

    return BettingGuide(
        primary_outcome=primary_name,
        primary_probability=primary_prob,
        confidence_tier=tier_zh,
        recommended_scoreline=recommended_score,
        recommended_total_goals=round(recommended_total, 2),
        over_25_pick=str(over_pick),
        over_25_probability=float(over_prob) if isinstance(over_prob, float) else 0.0,
        btts_pick=str(btts_pick),
        btts_probability=float(btts_prob) if isinstance(btts_prob, float) else 0.0,
        suggested_stake_pct=stake_pct,
        suggested_stake_usd=stake_usd,
        action_summary=action,
        rationale=rationale,
        cautions=cautions,
    )
