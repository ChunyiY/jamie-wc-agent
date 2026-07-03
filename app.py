"""Jamie — World Cup agent Streamlit frontend (fixtures · ML · player intel)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import APP_FULL_TITLE, APP_NAME, APP_TAGLINE
from src.betting_guide import build_betting_guide
from src.config import ensure_directories, load_config
from src.data_download import download_all, download_results, download_squad_data, download_worldcup_fixtures
from src.elo import EloSystem
from src.football_data import clean_fixture_field, load_fixtures_csv, load_results_csv
from src.match_analyzer import MatchAnalyzer, MatchPrediction
from src.model import FootballOutcomeModel
from src.model_loader import (
    ENSEMBLE_FILENAME,
    LoadSummary,
    load_or_train_football_stack,
    results_fingerprint,
)
from src.football_profile import (
    build_team_football_profile,
    profile_to_dataframe,
    tactical_detail_to_dataframe,
)
from src.match_stats import (
    MATCH_STATS_FILENAME,
    build_team_match_quality,
    load_intl_match_stats,
    quality_profile_to_dataframe,
)
from src.team_tactics import (
    build_club_tactical_proxy,
    build_international_tactical_profile,
    tactical_profile_to_dataframe,
)
from src.player_stats import (
    h2h_scorers_to_dataframe,
    head_to_head_scorers,
    load_goalscorers,
    players_to_dataframe,
    team_player_summary,
)
from src.model_validation import summarize_walk_forward
from src.prediction_stack import PredictionStack
from src.score_prediction import total_goals_summary
from src.squad_data import load_per90_csv, load_squads_csv, merge_squads_per90, team_squad_profile
from src.walk_forward import folds_to_dataframe
from src.visualizations import (
    calibration_chart,
    elo_comparison_chart,
    probability_comparison_chart,
    scoreline_heatmap_chart,
    top_scorelines_chart,
)

# ── 全局样式 ──────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0d9488 100%);
        padding: 1.6rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        color: #f8fafc;
    }
    .main-header h1 { color: #f8fafc !important; margin: 0; font-size: 2.1rem; letter-spacing: -0.02em; }
    .main-header .tagline { color: #5eead4; margin: 0.15rem 0 0; font-size: 1.05rem; font-weight: 600; }
    .main-header p  { color: #cbd5e1; margin: 0.5rem 0 0; font-size: 0.92rem; }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .disclaimer-box {
        background: #1c1917;
        border-left: 4px solid #f59e0b;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #d6d3d1;
        margin-bottom: 1rem;
    }
    .team-badge {
        display: inline-block;
        background: #0f766e;
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0 0.2rem;
    }
    div[data-testid="stSidebar"] { background: #0f172a; }
    div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
"""

DISCLAIMER = (
    "本工具仅供球迷研究与娱乐。预测基于历史国际比赛数据与滚动时间验证，"
    "过往表现不代表未来结果。不构成财务或博彩建议 — 下注前请对比真实赔率并量力而行。"
)

st.set_page_config(
    page_title=APP_FULL_TITLE,
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_directories()
CONFIG = load_config()
GOALSCORERS_PATH = CONFIG.goalscorers_csv


@st.cache_data(show_spinner=False)
def load_results(path_str: str) -> pd.DataFrame:
    return load_results_csv(Path(path_str))


@st.cache_data(show_spinner=False)
def load_goalscorers_cached(path_str: str) -> pd.DataFrame:
    return load_goalscorers(Path(path_str))


@st.cache_data(show_spinner=False)
def load_merged_squads(squads_path: str, per90_path: str) -> pd.DataFrame:
    squads = load_squads_csv(Path(squads_path))
    per90 = load_per90_csv(Path(per90_path))
    return merge_squads_per90(squads, per90)


# ── 模型加载 ──────────────────────────────────────────────────────────────────


def render_model_loading_ui(
    results: pd.DataFrame,
    *,
    force_retrain: bool = False,
    calibrate: bool | None = None,
) -> tuple[EloSystem, FootballOutcomeModel, MatchAnalyzer, LoadSummary]:
    progress_bar = st.progress(0, text="正在初始化 ML 管线…")
    status = st.empty()
    steps = ["Elo 评分", "特征 + Dixon–Coles", "集成模型", "就绪"]
    cols = st.columns(4)
    markers = [c.empty() for c in cols]

    def report(pct: float, message: str) -> None:
        progress_bar.progress(min(max(pct, 0.0), 1.0), text=message)
        status.markdown(f"**{message}**")
        if pct >= 0.25:
            markers[0].success(f"✓ {steps[0]}")
        if pct >= 0.78:
            markers[1].success(f"✓ {steps[1]}")
        if pct >= 0.95:
            markers[2].success(f"✓ {steps[2]}")
        if pct >= 1.0:
            markers[3].success(f"✓ {steps[3]}")

    stack = load_or_train_football_stack(
        results, CONFIG, force_retrain=force_retrain, calibrate=calibrate, progress=report
    )
    progress_bar.empty()
    status.empty()
    for m in markers:
        m.empty()
    return stack


def ensure_football_stack(
    results: pd.DataFrame,
) -> tuple[EloSystem, FootballOutcomeModel, MatchAnalyzer, LoadSummary | None]:
    fingerprint = results_fingerprint(results, CONFIG)
    force = bool(st.session_state.pop("force_retrain", False))
    calibrate = st.session_state.pop("train_calibrate", None)

    if (
        not force
        and st.session_state.get("stack_fingerprint") == fingerprint
        and "football_stack" in st.session_state
    ):
        elo, model, analyzer = st.session_state.football_stack
        return elo, model, analyzer, st.session_state.get("load_summary")

    elo, model, analyzer, summary = render_model_loading_ui(
        results, force_retrain=force, calibrate=calibrate
    )
    st.session_state.football_stack = (elo, model, analyzer)
    st.session_state.stack_fingerprint = fingerprint
    st.session_state.load_summary = summary
    return elo, model, analyzer, summary


# ── 工具函数 ──────────────────────────────────────────────────────────────────


def future_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty:
        return fixtures
    now = pd.Timestamp.now(tz="UTC")
    start_of_today = now.normalize()
    frame = fixtures.copy()
    dates = pd.to_datetime(frame["date"], utc=True)
    return frame.loc[dates.dt.normalize() >= start_of_today].sort_values("date").reset_index(drop=True)


def fixture_label(row: pd.Series) -> str:
    kickoff = pd.Timestamp(row["date"])
    if kickoff.tzinfo is None:
        kickoff = kickoff.tz_localize("UTC")
    stage = clean_fixture_field(row.get("stage")) or "小组赛"
    group = clean_fixture_field(row.get("group"))
    round_name = clean_fixture_field(row.get("round"))
    extra = group or round_name
    suffix = f" · {extra}" if extra else ""
    time_str = clean_fixture_field(row.get("time"))
    time_part = f" · {time_str}" if time_str else ""
    stage_zh = "淘汰赛" if str(row.get("stage", "")).lower() == "knockout" else "小组赛"
    return (
        f"{kickoff.strftime('%Y-%m-%d')} · "
        f"{row['team_a']} vs {row['team_b']} · {stage_zh}{suffix}{time_part}"
    )


def fixtures_table(upcoming: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in upcoming.iterrows():
        kickoff = pd.Timestamp(row["date"])
        if kickoff.tzinfo is None:
            kickoff = kickoff.tz_localize("UTC")
        rows.append(
            {
                "日期": kickoff.strftime("%Y-%m-%d"),
                "主队": row["team_a"],
                "客队": row["team_b"],
                "小组": clean_fixture_field(row.get("group")) or "—",
                "阶段": "淘汰赛" if str(row.get("stage", "")).lower() == "knockout" else "小组赛",
                "球场": clean_fixture_field(row.get("ground")) or "待定",
                "开球": clean_fixture_field(row.get("time")) or "—",
            }
        )
    return pd.DataFrame(rows)


def _ensemble_path() -> Path:
    if hasattr(CONFIG, "models_dir"):
        p = CONFIG.models_dir / ENSEMBLE_FILENAME
        if p.exists():
            return p
    return PROJECT_ROOT / "models" / ENSEMBLE_FILENAME


# ── 侧边栏 ────────────────────────────────────────────────────────────────────


def sidebar(results: pd.DataFrame, model: FootballOutcomeModel, summary: LoadSummary | None) -> float:
    st.sidebar.markdown(f"## ⚽ {APP_NAME}")
    st.sidebar.caption(APP_TAGLINE)
    bankroll = st.sidebar.number_input("资金池 ($)", min_value=10.0, value=70.0, step=5.0)
    st.sidebar.markdown(f"历史比赛：**{len(results):,}** 场")
    if summary:
        st.sidebar.success(f"模型就绪 · {summary.teams} 支球队")
        w = summary.ensemble_weights
        st.sidebar.markdown(
            f"**集成权重**  \n"
            f"Dixon–Coles {w[0]:.0%}  \n"
            f"有序 Logit {w[1]:.0%}  \n"
            f"多项 Logit {w[2]:.0%}"
        )
    if model.bundle:
        m = model.bundle.metrics
        st.sidebar.markdown("**回测指标**")
        st.sidebar.write(f"对数损失：{m.log_loss:.3f}")
        st.sidebar.write(f"Brier：{m.brier_score:.3f}")
        st.sidebar.write(f"准确率：{m.accuracy:.1%}")

    with st.sidebar.expander("模型与数据"):
        if st.button("快速重训", key="sb_fast", use_container_width=True):
            st.session_state.force_retrain = True
            st.session_state.train_calibrate = False
            st.session_state.pop("football_stack", None)
            st.rerun()
        if st.button("完整重训 + 校准", key="sb_full", use_container_width=True):
            st.session_state.force_retrain = True
            st.session_state.train_calibrate = True
            st.session_state.pop("football_stack", None)
            st.rerun()
        if st.button("下载最新数据", key="sb_dl", use_container_width=True):
            with st.spinner("正在下载比赛、进球、点球、名单、WC2026 xG 与 StatsBomb 国际赛数据…"):
                download_all(prefer_kaggle=True)
            load_results.clear()
            load_goalscorers_cached.clear()
            load_merged_squads.clear()
            st.session_state.force_retrain = True
            st.rerun()
    return float(bankroll)


# ── 球员情报 ──────────────────────────────────────────────────────────────────


def render_tactical_panel(
    results: pd.DataFrame,
    goalscorers: pd.DataFrame,
    merged_squads: pd.DataFrame,
    intl_match_stats: pd.DataFrame,
    elo: EloSystem,
    team_a: str,
    team_b: str,
    *,
    match_date: pd.Timestamp,
) -> None:
    """Team tactics: international mentality + xG/SOT + discounted club proxies."""
    with st.expander("📊 战术与球队层面分析", expanded=True):
        st.caption(
            "国际赛：历史战绩 + StatsBomb/WC2026 的 xG·射正率；俱乐部 per-90 仅弱信号（已降权）。"
        )
        col_a, col_b = st.columns(2)
        for col, team in ((col_a, team_a), (col_b, team_b)):
            with col:
                st.markdown(f"#### {team}")
                if not intl_match_stats.empty:
                    mq = build_team_match_quality(
                        intl_match_stats, team, match_date, prediction_context="world_cup"
                    )
                    if mq.sample_matches > 0:
                        st.markdown("**比赛质量 (xG / 射正)**")
                        st.dataframe(quality_profile_to_dataframe(mq), use_container_width=True, hide_index=True)
                intl = build_international_tactical_profile(
                    results, goalscorers, team, match_date, elo, prediction_context="world_cup"
                )
                st.markdown("**国家队层面**")
                st.dataframe(tactical_profile_to_dataframe(intl), use_container_width=True, hide_index=True)
                club = build_club_tactical_proxy(merged_squads, team)
                if club:
                    st.markdown("**俱乐部 proxy（已降权 ~72%）**")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"指标": "锋线射门/90", "数值": f"{club.shot_volume:.2f}"},
                                {"指标": "射门转化率 proxy", "数值": f"{club.shot_conversion:.1%}"},
                                {"指标": "传球成功率", "数值": f"{club.pass_accuracy:.1f}%"},
                                {"指标": "防守动作/90", "数值": f"{club.defensive_actions:.2f}"},
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                fp = build_team_football_profile(merged_squads, team)
                if fp:
                    st.markdown(f"**风格** {fp.style_label} · **阵型** {fp.formation_hint}")
                    st.caption("阵型由估值首发 XI 按位置槽位拟合（1 门将 + 标准模板），非单纯全队 Top11。")
                    st.dataframe(profile_to_dataframe(fp), use_container_width=True, hide_index=True)
                    st.markdown("**预计首发结构**")
                    st.dataframe(tactical_detail_to_dataframe(fp), use_container_width=True, hide_index=True)


def render_player_panel(
    goalscorers: pd.DataFrame,
    merged_squads: pd.DataFrame,
    team_a: str,
    team_b: str,
    *,
    match_date: pd.Timestamp | None = None,
    expanded: bool = True,
) -> None:
    ref = match_date or pd.Timestamp.now(tz="UTC")
    with st.expander("👤 球员与阵容情报", expanded=expanded):
        col_a, col_b = st.columns(2)
        for col, team in ((col_a, team_a), (col_b, team_b)):
            summary = team_player_summary(
                goalscorers, team, merged_squads=merged_squads, reference_date=ref, top_n=10
            )
            profile = summary.wc_squad
            with col:
                st.markdown(f"#### {team}")
                if profile:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("世界杯名单", profile.squad_size)
                    m2.metric("TOP11 RT估值", f"€{profile.top11_market_value_eur/1e6:.0f}M", help="RisingTransfers 模型估算，非 Transfermarkt 实时价")
                    m3.metric("锋线进球/90", f"{profile.top11_goals_per90:.2f}", help="TOP5 锋线球员俱乐部场均进球")
                    m4.metric("平均年龄", f"{profile.avg_age:.1f}")
                    st.caption(
                        "RT估值来自 RisingTransfers 算法（老将/MLS 常被低估，≠ Transfermarkt 实时价）。"
                    )
                elif summary.active_players == 0:
                    st.info("暂无现役球员数据，请先下载世界杯名单。")
                    continue
                else:
                    st.warning("未匹配世界杯名单，仅显示近期国家队进球者。")
                st.markdown("**当前贡献 TOP（俱乐部 per-90 + 国家队近12月）**")
                st.dataframe(
                    players_to_dataframe(summary.top_contributors),
                    use_container_width=True,
                    hide_index=True,
                )

        h2h = head_to_head_scorers(goalscorers, team_a, team_b)
        st.markdown("#### 近15年交锋射手")
        if h2h:
            st.dataframe(h2h_scorers_to_dataframe(h2h), use_container_width=True, hide_index=True)
        else:
            st.caption("两队暂无交锋进球记录。")


def render_player_explorer(
    goalscorers: pd.DataFrame,
    merged_squads: pd.DataFrame,
    upcoming: pd.DataFrame,
) -> None:
    st.markdown("### 👤 球员情报中心")
    st.caption(
        "世界杯 2026 名单（risingtransfers）+ 俱乐部 per-90 数据 + 国家队近期进球记录。"
        "默认按「贡献指数」排序，已过滤久未入选球员。"
    )

    teams = sorted(set(upcoming["team_a"]) | set(upcoming["team_b"])) if not upcoming.empty else []
    if not teams and not merged_squads.empty:
        teams = sorted(merged_squads["country"].unique())
    if not teams:
        teams = sorted(goalscorers["team"].unique())[:48]

    team = st.selectbox("选择球队", teams, index=0)
    summary = team_player_summary(goalscorers, team, merged_squads=merged_squads, top_n=25)
    profile = summary.wc_squad

    c1, c2, c3, c4, c5 = st.columns(5)
    if profile:
        c1.metric("名单人数", profile.squad_size)
        c2.metric("总身价", f"€{profile.total_market_value_eur/1e6:.0f}M")
        c3.metric("TOP11 进球/90", f"{profile.top11_goals_per90:.2f}")
        c4.metric("进攻深度", profile.attack_depth)
        c5.metric("现役贡献者", summary.active_players)
    else:
        c1.metric("现役球员", summary.active_players)
        c2.caption("下载世界杯名单以查看身价与 per-90")

    tab_squad, tab_chart = st.tabs(["阵容详情", "贡献指数"])

    with tab_squad:
        st.dataframe(players_to_dataframe(summary.top_contributors), use_container_width=True, hide_index=True)

    with tab_chart:
        if summary.top_contributors:
            chart_df = pd.DataFrame(
                {
                    "球员": [r.name for r in summary.top_contributors[:12]],
                    "贡献指数": [r.contribution_score for r in summary.top_contributors[:12]],
                }
            )
            fig = px.bar(
                chart_df, x="球员", y="贡献指数",
                title=f"{team} — 球员贡献指数 TOP 12",
                color="贡献指数", color_continuous_scale="Teal",
            )
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("暂无数据，请先下载世界杯名单。")


# ── 模型实验室 ────────────────────────────────────────────────────────────────


def render_model_panel(model: FootballOutcomeModel, summary: LoadSummary | None) -> None:
    st.markdown("### 🧪 模型实验室")
    st.caption("**主指标**：Walk-forward 对数损失与 RPS（滚动时间验证）。单次留出准确率仅供参考。")
    if model.bundle is None:
        st.warning("模型尚未训练。")
        return

    meta_path = PROJECT_ROOT / "models" / "model_meta.json"
    baseline = {}
    if meta_path.exists():
        try:
            baseline = json.loads(meta_path.read_text(encoding="utf-8")).get("baseline", {})
        except json.JSONDecodeError:
            baseline = {}

    ensemble_path = _ensemble_path()
    wf_summary = {}
    if ensemble_path.exists():
        stack = PredictionStack.load(ensemble_path)
        if stack.walk_forward_folds:
            wf_summary = summarize_walk_forward(folds_to_dataframe(stack.walk_forward_folds))

    if wf_summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("★ 集成 Log-loss (WF)", f"{wf_summary.get('avg_ensemble_log_loss', 0):.3f}", help="Walk-forward 平均，越低越好")
        c2.metric("★ 集成 RPS (WF)", f"{wf_summary.get('avg_ensemble_rps', 0):.3f}")
        c3.metric("集成准确率 (WF)", f"{wf_summary.get('avg_ensemble_accuracy', 0):.1%}")
        c4.metric("Walk-forward 折数", int(summary.walk_forward_folds if summary else 0))
    else:
        st.info("运行「完整重训 + 校准」以生成 Walk-forward 验证结果（推荐以此评估模型）。")

    st.markdown("#### 与基线对比（单次时间留出）")
    m = model.bundle.metrics
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("模型 Log-loss", f"{m.log_loss:.3f}")
    b2.metric("模型准确率", f"{m.accuracy:.1%}")
    if baseline:
        b3.metric("Elo 基线 Log-loss", f"{baseline.get('elo_log_loss', 0):.3f}")
        b4.metric("Elo 基线准确率", f"{baseline.get('elo_accuracy', 0):.1%}")
        delta_ll = baseline.get("log_loss_vs_elo", 0)
        delta_acc = baseline.get("accuracy_vs_elo", 0)
        b5.metric("vs Elo ΔLog-loss", f"{delta_ll:+.3f}", help="正数=模型更好")
        b6.metric("vs Elo Δ准确率", f"{delta_acc:+.1%}")
    else:
        b3.metric("Brier", f"{m.brier_score:.3f}")
        b4.metric("校准", "开启" if summary and summary.calibrated else "关闭")

    if summary:
        w = summary.ensemble_weights
        st.markdown(
            f"**集成权重** — Dixon–Coles **{w[0]:.0%}** · "
            f"有序 Logit **{w[1]:.0%}** · 多项 Logit **{w[2]:.0%}** · "
            f"**阵容特征** 已接入（世界杯名单 + 近期进球者）"
        )

    st.markdown("#### 方法论")
    st.markdown(
        """
| 组件 | 作用 |
|------|------|
| **Dixon–Coles (1997)** | 泊松比分矩阵 + 低比分 ρ 修正；样本按赛事层级加权 |
| **Rue & Salvesen (2000)** | 时间衰减：近期国际赛权重更高 |
| **Baio & Blangiardo (2010)** | 赛事分层：世界杯/预选赛 > 友谊赛 |
| **StatsBomb Open Data** | 国际大赛事件级 xG、射门、射正（2018–2024） |
| **mominullptr WC2026** | 2026 世界杯预选赛/友谊赛 xG、控球、FIFA 排名 |
| **martj42 shootouts** | 历史点球大战 → 淘汰赛晋级概率 |
| **有序 Logit** | 平局感知的 W/D/L |
| **多项 Logit** | Elo + 状态 + **战术/心态** 国际赛特征 |
| **Walk-forward** | 滚动验证，防止过拟合 |
        """
    )

    if m.calibration_mean_predicted:
        st.plotly_chart(
            calibration_chart(m.calibration_fractions, m.calibration_mean_predicted, lang="zh"),
            use_container_width=True,
        )

    if ensemble_path.exists() and wf_summary:
        stack = PredictionStack.load(ensemble_path)
        if stack.walk_forward_folds:
            st.markdown("#### 滚动时间验证明细")
            wf_df = folds_to_dataframe(stack.walk_forward_folds)
            wf_display = wf_df.rename(
                columns={
                    "fold": "折",
                    "n_train": "训练场次",
                    "n_test": "测试场次",
                    "log_loss_ensemble": "集成对数损失",
                    "log_loss_dixon_coles": "DC对数损失",
                    "log_loss_ordered": "OL对数损失",
                    "log_loss_multinomial": "ML对数损失",
                    "accuracy_ensemble": "集成准确率",
                    "rps_ensemble": "集成RPS",
                }
            )
            st.dataframe(wf_display, use_container_width=True, hide_index=True)


# ── 比赛分析报告 ──────────────────────────────────────────────────────────────


def render_form_comparison(prediction: MatchPrediction) -> None:
    fa, fb = prediction.form_a, prediction.form_b
    st.markdown("#### 近期战绩")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "球队": prediction.team_a,
                    "近5场 W-D-L": f"{fa.wins5}-{fa.draws5}-{fa.losses5}",
                    "近10场 W-D-L": f"{fa.wins10}-{fa.draws10}-{fa.losses10}",
                    "场均进球": f"{fa.goals_for_avg:.2f}",
                    "场均失球": f"{fa.goals_against_avg:.2f}",
                    "对手均 Elo": f"{fa.opponent_elo_avg:.0f}",
                },
                {
                    "球队": prediction.team_b,
                    "近5场 W-D-L": f"{fb.wins5}-{fb.draws5}-{fb.losses5}",
                    "近10场 W-D-L": f"{fb.wins10}-{fb.draws10}-{fb.losses10}",
                    "场均进球": f"{fb.goals_for_avg:.2f}",
                    "场均失球": f"{fb.goals_against_avg:.2f}",
                    "对手均 Elo": f"{fb.opponent_elo_avg:.0f}",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_prediction_report(
    prediction: MatchPrediction,
    guide,
    elo: EloSystem,
    results: pd.DataFrame,
    goalscorers: pd.DataFrame,
    merged_squads: pd.DataFrame,
    intl_match_stats: pd.DataFrame,
    *,
    match_date: pd.Timestamp | None = None,
) -> None:
    probs = {
        "team_a": prediction.prob_team_a_win,
        "draw": prediction.prob_draw,
        "team_b": prediction.prob_team_b_win,
    }

    st.markdown("---")
    st.markdown(
        f"## <span class='team-badge'>{prediction.team_a}</span> "
        f"vs <span class='team-badge'>{prediction.team_b}</span>",
        unsafe_allow_html=True,
    )

    o1, o2, o3, o4 = st.columns(4)
    o1.metric(f"{prediction.team_a} 胜", f"{prediction.prob_team_a_win:.1%}")
    o2.metric("平局", f"{prediction.prob_draw:.1%}")
    o3.metric(f"{prediction.team_b} 胜", f"{prediction.prob_team_b_win:.1%}")
    o4.metric("模型置信度", f"{prediction.confidence:.1%}")

    if prediction.uncertainty:
        u = prediction.uncertainty
        st.caption(
            f"不确定性区间（约 90%）：{prediction.team_a} ±{u[0]:.1%} · "
            f"平局 ±{u[1]:.1%} · {prediction.team_b} ±{u[2]:.1%}"
        )

    if prediction.component_probs:
        comp = prediction.component_probs
        st.markdown("#### 集成子模型分解")
        comp_df = pd.DataFrame(
            [
                {
                    "子模型": "Dixon–Coles",
                    f"{prediction.team_a} 胜": f"{comp.dixon_coles[0]:.1%}",
                    "平局": f"{comp.dixon_coles[1]:.1%}",
                    f"{prediction.team_b} 胜": f"{comp.dixon_coles[2]:.1%}",
                },
                {
                    "子模型": "有序 Logit",
                    f"{prediction.team_a} 胜": f"{comp.ordered_logit[0]:.1%}",
                    "平局": f"{comp.ordered_logit[1]:.1%}",
                    f"{prediction.team_b} 胜": f"{comp.ordered_logit[2]:.1%}",
                },
                {
                    "子模型": "多项 Logit",
                    f"{prediction.team_a} 胜": f"{comp.multinomial[0]:.1%}",
                    "平局": f"{comp.multinomial[1]:.1%}",
                    f"{prediction.team_b} 胜": f"{comp.multinomial[2]:.1%}",
                },
            ]
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("#### 比分与进球预测")
    if prediction.score_prediction:
        sp = prediction.score_prediction

        tg = total_goals_summary(sp.score_matrix, expected_goals_a=sp.expected_goals_a, expected_goals_b=sp.expected_goals_b)

        st.info(
            f"**为何「大 2.5」与「{sp.most_likely.label}」可同时出现？**  "
            f"「{sp.most_likely.label}」是**某一个精确比分**的最高概率（仅 {sp.most_likely.probability:.1%}），"
            f"并不等于整场比赛只会有这一种结果。"
            f" 预期总进球 **{tg.expected_total:.2f}** 球，"
            f"**总进球最可能是 {tg.most_likely_total} 球**（{tg.most_likely_total_prob:.1%}）。"
            f" 所有 ≥3 球的比分加总 → 大 2.5 为 **{tg.over_25_prob:.1%}**。"
        )

        g1, g2, g3, g4, g5, g6 = st.columns(6)
        g1.metric(
            "最热门精确比分",
            sp.most_likely.label,
            f"单格 {sp.most_likely.probability:.1%}",
            help="36 种精确比分里概率最高的那一格，通常只有 8–12%",
        )
        g2.metric(f"{prediction.team_a} xG", f"{sp.expected_goals_a:.2f}")
        g3.metric(f"{prediction.team_b} xG", f"{sp.expected_goals_b:.2f}")
        g4.metric("预期总进球", f"{tg.expected_total:.2f}", help="xG 之和，可理解为平均进球数")
        g5.metric(
            "最可能总进球",
            str(tg.most_likely_total),
            f"{tg.most_likely_total_prob:.1%}",
            help="把所有精确比分按总进球合并后的众数",
        )
        g6.metric("大 2.5", f"{tg.over_25_prob:.1%}", help="总进球≥3 的所有比分概率之和")

        btts_col, under_col = st.columns(2)
        btts_col.metric("双方进球 BTTS", f"{sp.btts_prob:.1%}")
        under_col.metric("小 2.5", f"{tg.under_25_prob:.1%}")

        totals_text = " · ".join(f"{t}球 {p:.1%}" for t, p in tg.top_totals[:4])
        st.caption(f"总进球分布 TOP：{totals_text}")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                top_scorelines_chart(prediction.team_a, prediction.team_b, sp.top_scorelines, lang="zh"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                scoreline_heatmap_chart(
                    prediction.team_a,
                    prediction.team_b,
                    sp.score_matrix,
                    most_likely=(sp.most_likely.goals_a, sp.most_likely.goals_b),
                    lang="zh",
                ),
                use_container_width=True,
            )

    render_tactical_panel(
        results,
        goalscorers,
        merged_squads,
        intl_match_stats,
        elo,
        prediction.team_a,
        prediction.team_b,
        match_date=match_date or pd.Timestamp.now(tz="UTC"),
    )

    render_player_panel(
        goalscorers, merged_squads, prediction.team_a, prediction.team_b, expanded=True
    )

    st.markdown("#### 观赛指南（仅模型概率，不含市场赔率）")
    st.info(guide.action_summary)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("胜平负倾向", guide.primary_outcome, f"{guide.primary_probability:.1%}")
    b2.metric("进球倾向", guide.over_25_pick, f"{guide.over_25_probability:.1%}")
    b3.metric("BTTS 倾向", guide.btts_pick, f"{guide.btts_probability:.1%}")
    b4.metric("仓位上限", f"${guide.suggested_stake_usd:.2f}", f"{guide.confidence_tier}置信")

    with st.expander("分析依据与风险提示"):
        st.markdown("**模型依据**")
        for line in guide.rationale:
            st.markdown(f"- {line}")
        st.markdown("**技术说明**")
        for line in prediction.explanation:
            st.markdown(f"- {line}")
        st.markdown("**风险提示**")
        for line in guide.cautions:
            st.markdown(f"- {line}")

    st.plotly_chart(
        probability_comparison_chart(prediction.team_a, prediction.team_b, probs, lang="zh"),
        use_container_width=True,
    )
    st.plotly_chart(
        elo_comparison_chart(elo.to_dataframe(), prediction.team_a, prediction.team_b, lang="zh"),
        use_container_width=True,
    )
    render_form_comparison(prediction)


# ── 主入口 ────────────────────────────────────────────────────────────────────


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        f'<div class="main-header"><h1>⚽ {APP_NAME}</h1>'
        f'<p class="tagline">{APP_TAGLINE}</p>'
        "<p>Dixon–Coles · 有序 Logit · 多项 Logit · 球员情报 · 2026 世界杯赛程</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="disclaimer-box">{DISCLAIMER}</div>', unsafe_allow_html=True)

    results = load_results(str(CONFIG.results_csv))
    goalscorers = load_goalscorers_cached(str(GOALSCORERS_PATH))
    merged_squads = load_merged_squads(str(CONFIG.wc_squads_csv), str(CONFIG.wc_per90_csv))
    data_dir = CONFIG.results_csv.parent
    intl_match_stats = load_intl_match_stats(data_dir / MATCH_STATS_FILENAME)
    elo, model, analyzer, summary = ensure_football_stack(results)
    bankroll = sidebar(results, model, summary)

    all_fixtures = load_fixtures_csv(CONFIG.fixtures_csv)
    upcoming = future_fixtures(all_fixtures)

    tab_schedule, tab_predict, tab_players, tab_model = st.tabs(
        ["📅 赛程中心", "🔮 比赛分析", "👤 球员情报", "🧪 模型实验室"]
    )

    with tab_schedule:
        st.markdown("### 📅 未来赛程")
        if upcoming.empty:
            st.warning("暂无未来赛程。请在侧边栏点击「下载最新数据」刷新 2026 世界杯赛程。")
        else:
            groups = ["全部"] + sorted(
                {clean_fixture_field(r.get("group")) for _, r in upcoming.iterrows() if clean_fixture_field(r.get("group"))}
            )
            group_filter = st.selectbox("按小组筛选", groups, index=0)
            shown = upcoming if group_filter == "全部" else upcoming[
                upcoming["group"].map(clean_fixture_field) == group_filter
            ]
            st.metric("待赛场次", len(shown))
            st.dataframe(fixtures_table(shown), use_container_width=True, hide_index=True)

    with tab_players:
        render_player_explorer(goalscorers, merged_squads, upcoming)

    with tab_model:
        render_model_panel(model, summary)

    with tab_predict:
        st.markdown("### 🔮 比赛 ML 分析")
        if upcoming.empty:
            st.warning("暂无未来赛程可分析。")
            return

        labels = [fixture_label(upcoming.iloc[i]) for i in range(len(upcoming))]
        choice = st.selectbox("选择比赛", labels, index=0)
        row = upcoming.iloc[labels.index(choice)]

        kickoff = pd.Timestamp(row["date"])
        if kickoff.tzinfo is None:
            kickoff = kickoff.tz_localize("UTC")

        meta1, meta2, meta3, meta4 = st.columns(4)
        meta1.write(f"**开球时间**  \n{kickoff.strftime('%Y-%m-%d %H:%M UTC')}")
        round_name = clean_fixture_field(row.get("round"))
        group_name = clean_fixture_field(row.get("group"))
        stage_detail = group_name or round_name or "—"
        stage_zh = "淘汰赛" if str(row.get("stage", "")).lower() == "knockout" else "小组赛"
        meta2.write(f"**阶段**  \n{stage_zh} · {stage_detail}")
        meta3.write(f"**球场**  \n{clean_fixture_field(row.get('ground')) or '待定'}")
        meta4.write(f"**赛事**  \n{row.get('tournament', 'FIFA World Cup')}")

        render_player_panel(
            goalscorers, merged_squads, row["team_a"], row["team_b"],
            match_date=kickoff, expanded=False,
        )

        knockout = str(row.get("stage", "group")).lower() == "knockout"
        analyze = st.button("开始 ML 分析", type="primary", use_container_width=True)

        if analyze:
            with st.spinner("正在运行集成模型…"):
                prediction = analyzer.predict_match(
                    row["team_a"],
                    row["team_b"],
                    match_date=kickoff,
                    neutral=bool(row.get("neutral", True)),
                    tournament=str(row.get("tournament", "FIFA World Cup")),
                    knockout=knockout,
                    stage=str(row.get("stage", "group")),
                )
                guide = build_betting_guide(prediction, model=model, bankroll=bankroll)
            render_prediction_report(
                prediction, guide, elo, results, goalscorers, merged_squads, intl_match_stats, match_date=kickoff
            )


if __name__ == "__main__":
    main()
