"""Plotly visualizations for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def elo_comparison_chart(elo_table: pd.DataFrame, team_a: str, team_b: str, *, lang: str = "en") -> go.Figure:
    subset = elo_table[elo_table["team"].isin([team_a, team_b])].copy()
    if subset.empty:
        subset = pd.DataFrame({"team": [team_a, team_b], "elo": [1500, 1500]})
    title = "Elo 评分对比" if lang == "zh" else "Elo Rating Comparison"
    y_label = "Elo 分" if lang == "zh" else "Elo"
    x_label = "球队" if lang == "zh" else "Team"
    fig = px.bar(subset, x="team", y="elo", title=title, text="elo")
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(yaxis_title=y_label, xaxis_title=x_label)
    return fig


def probability_comparison_chart(
    team_a: str,
    team_b: str,
    model_probs: dict[str, float],
    market_prob: float | None = None,
    *,
    lang: str = "en",
) -> go.Figure:
    if lang == "zh":
        labels = [f"{team_a} 胜", "平局", f"{team_b} 胜"]
        title = "胜平负概率"
        y_label = "概率"
        model_name = "集成模型"
    else:
        labels = [f"{team_a} win", "Draw", f"{team_b} win"]
        title = "Model vs Market Probability"
        y_label = "Probability"
        model_name = "Model"
    values = [model_probs["team_a"], model_probs["draw"], model_probs["team_b"]]
    fig = go.Figure(
        data=[
            go.Bar(name=model_name, x=labels, y=values),
        ]
    )
    if market_prob is not None:
        fig.add_trace(go.Bar(name="Market (YES proxy)", x=[labels[0]], y=[market_prob]))
    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        barmode="group",
        yaxis=dict(tickformat=".0%"),
    )
    return fig


def calibration_chart(fractions: list[float], mean_predicted: list[float], *, lang: str = "en") -> go.Figure:
    title = "模型校准曲线" if lang == "zh" else "Model Calibration Curve"
    x_label = "预测概率均值" if lang == "zh" else "Mean predicted probability"
    y_label = "实际命中率" if lang == "zh" else "Fraction of positives"
    perfect = "完美校准" if lang == "zh" else "Perfect calibration"
    cal_name = "校准" if lang == "zh" else "Calibration"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mean_predicted,
            y=fractions,
            mode="lines+markers",
            name=cal_name,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name=perfect,
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    return fig


def edge_distribution_chart(edges: list[float]) -> go.Figure:
    fig = px.histogram(
        x=edges,
        nbins=20,
        title="Adjusted Edge Distribution",
        labels={"x": "Adjusted edge"},
    )
    fig.add_vline(x=0.06, line_dash="dash", annotation_text="Default threshold")
    return fig


def paper_pnl_chart(paper_trades: pd.DataFrame) -> go.Figure:
    if paper_trades.empty:
        return go.Figure().update_layout(title="Paper Trading PnL (no trades yet)")
    frame = paper_trades.copy()
    frame["cumulative_pnl"] = frame["pnl"].fillna(0).cumsum()
    fig = px.line(frame, x="created_at", y="cumulative_pnl", title="Paper Trading Cumulative PnL")
    return fig


def scoreline_heatmap_chart(
    team_a: str,
    team_b: str,
    score_matrix: list[list[float]],
    *,
    most_likely: tuple[int, int] | None = None,
    lang: str = "en",
) -> go.Figure:
    """Heatmap of exact scoreline probabilities (team A goals × team B goals)."""
    import numpy as np

    z = np.array(score_matrix)
    goals = list(range(z.shape[0]))
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[f"{g}" for g in goals],
            y=[f"{g}" for g in goals],
            colorscale="Blues",
            text=[[f"{val:.1%}" if val >= 0.03 else "" for val in row] for row in z],
            texttemplate="%{text}",
            hovertemplate=(
                f"{team_a}: %{{y}} goals<br>{team_b}: %{{x}} goals<br>"
                "Probability: %{z:.1%}<extra></extra>"
            ),
        )
    )
    if most_likely is not None:
        ha, hb = most_likely
        fig.add_annotation(
            x=str(hb),
            y=str(ha),
            text="★",
            showarrow=False,
            font=dict(size=18, color="orange"),
        )
    title = "精确比分概率热力图" if lang == "zh" else "Exact Scoreline Probabilities (Poisson)"
    x_title = f"{team_b} 进球" if lang == "zh" else f"{team_b} goals"
    y_title = f"{team_a} 进球" if lang == "zh" else f"{team_a} goals"
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def top_scorelines_chart(team_a: str, team_b: str, scorelines: list, *, lang: str = "en") -> go.Figure:
    labels = [f"{s.goals_a}–{s.goals_b}" for s in scorelines]
    probs = [s.probability for s in scorelines]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=probs,
                text=[f"{p:.1%}" for p in probs],
                textposition="outside",
            )
        ]
    )
    title = f"最热门精确比分 TOP ({team_a} vs {team_b})" if lang == "zh" else f"Top Likely Scores ({team_a} – {team_b})"
    x_label = "比分" if lang == "zh" else "Scoreline"
    y_label = "概率" if lang == "zh" else "Probability"
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis=dict(tickformat=".0%"),
    )
    return fig
