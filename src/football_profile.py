"""Team-level football profile: formation hint, attack/defense/midfield/GK indices."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.team_mapping import canonical_team_name

POS_GK = {"GK"}
POS_DEF = {"DF", "CB", "LB", "RB", "WB"}
POS_MID = {"MF", "CM", "DM", "AM", "WG", "LM", "RM"}
POS_ATT = {"FW", "ST", "CF", "SS"}

# (label, defenders, midfielders, attackers) — outfield slots sum to 10.
FORMATION_TEMPLATES: list[tuple[str, int, int, int]] = [
    ("4-3-3", 4, 3, 3),
    ("4-4-2", 4, 4, 2),
    ("4-2-3-1", 4, 5, 1),
    ("3-5-2", 3, 5, 2),
    ("3-4-3", 3, 4, 3),
    ("5-3-2", 5, 3, 2),
    ("5-4-1", 5, 4, 1),
    ("4-1-4-1", 4, 5, 1),
]


def _pos_group(pos: str) -> str:
    p = str(pos).upper().strip()
    if p in POS_GK:
        return "GK"
    if p in POS_DEF or p.startswith("D"):
        return "DEF"
    if p in POS_ATT or p.startswith("F"):
        return "ATT"
    return "MID"


def _value_col(frame: pd.DataFrame) -> str:
    if "rt_value_estimate_eur" in frame.columns:
        return "rt_value_estimate_eur"
    if "market_value_eur" in frame.columns:
        return "market_value_eur"
    return frame.columns[0]


def _col_mean(frame: pd.DataFrame, col: str) -> float:
    if col not in frame.columns or frame.empty:
        return 0.0
    return float(pd.to_numeric(frame[col], errors="coerce").fillna(0).mean())


def _pick_top(frame: pd.DataFrame, n: int, val_col: str) -> pd.DataFrame:
    if frame.empty or n <= 0:
        return frame.iloc[0:0]
    return frame.nlargest(min(n, len(frame)), val_col)


def _refine_midfield_shape(label: str, mids: pd.DataFrame) -> str:
    """Distinguish 4-2-3-1 vs 4-1-4-1 when both use 4-5-1 slots."""
    if label not in {"4-2-3-1", "4-1-4-1"} or mids.empty:
        return label
    val_col = _value_col(mids)
    ordered = mids.nlargest(min(5, len(mids)), val_col)
    if len(ordered) < 3:
        return label
    top2_tackles = _col_mean(ordered.head(2), "tackles_per90")
    rest_key_passes = _col_mean(ordered.tail(max(1, len(ordered) - 2)), "key_passes_per90")
    if top2_tackles >= 1.8 and rest_key_passes >= 1.0:
        return "4-2-3-1"
    if _col_mean(ordered, "pass_accuracy_pct") >= 84:
        return "4-1-4-1"
    return "4-2-3-1"


def select_starting_xi(subset: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    Pick a realistic XI: 1 GK + best-fit outfield template by squad value.

    Avoids the old bug of taking global top-11 (which put 2 GKs and 6 CBs for Spain).
    """
    if subset.empty:
        return "未知", subset

    val_col = _value_col(subset)
    subset = subset.copy()
    subset["_pos_group"] = subset["position"].astype(str).map(_pos_group)

    gk_pool = subset[subset["_pos_group"] == "GK"]
    gk = _pick_top(gk_pool, 1, val_col)
    if gk.empty:
        return "未知", subset.head(0)

    def_pool = subset[subset["_pos_group"] == "DEF"]
    mid_pool = subset[subset["_pos_group"] == "MID"]
    att_pool = subset[subset["_pos_group"] == "ATT"]

    best_label = "4-3-3"
    best_score = -1.0
    best_parts: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] = (
        def_pool.head(0),
        mid_pool.head(0),
        att_pool.head(0),
    )

    for label, n_def, n_mid, n_att in FORMATION_TEMPLATES:
        if len(def_pool) < n_def or len(mid_pool) < n_mid or len(att_pool) < n_att:
            continue
        defs = _pick_top(def_pool, n_def, val_col)
        mids = _pick_top(mid_pool, n_mid, val_col)
        atts = _pick_top(att_pool, n_att, val_col)
        score = (
            float(pd.to_numeric(defs[val_col], errors="coerce").fillna(0).sum())
            + float(pd.to_numeric(mids[val_col], errors="coerce").fillna(0).sum())
            + float(pd.to_numeric(atts[val_col], errors="coerce").fillna(0).sum())
        )
        if score > best_score:
            best_score = score
            best_label = label
            best_parts = (defs, mids, atts)

    defs, mids, atts = best_parts
    if defs.empty and mids.empty and atts.empty:
        # Fallback: 4-3-3 with whatever positions exist
        outfield = subset[subset["_pos_group"] != "GK"].nlargest(10, val_col)
        xi = pd.concat([gk, outfield], ignore_index=True)
        return infer_formation_from_xi(xi), xi

    best_label = _refine_midfield_shape(best_label, mids)
    xi = pd.concat([gk, defs, mids, atts], ignore_index=True)
    return best_label, xi


def infer_formation_from_xi(xi: pd.DataFrame) -> str:
    """Count position groups in a selected XI and snap to nearest canonical formation."""
    if xi.empty:
        return "未知"
    groups = xi["position"].astype(str).map(_pos_group) if "position" in xi.columns else pd.Series(dtype=str)
    n_def = int((groups == "DEF").sum())
    n_mid = int((groups == "MID").sum())
    n_att = int((groups == "ATT").sum())

    best_label = "4-3-3"
    best_dist = 999
    for label, td, tm, ta in FORMATION_TEMPLATES:
        dist = abs(n_def - td) + abs(n_mid - tm) + abs(n_att - ta)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def infer_formation(top11: pd.DataFrame) -> str:
    """Backward-compatible wrapper — prefer select_starting_xi for new code."""
    label, _ = select_starting_xi(top11)
    return label


def _tactical_breakdown(
    def_line: pd.DataFrame,
    mid_line: pd.DataFrame,
    att_line: pd.DataFrame,
) -> dict[str, str]:
    pass_acc = _col_mean(mid_line, "pass_accuracy_pct")
    key_passes = _col_mean(mid_line, "key_passes_per90")
    shots = _col_mean(att_line, "shots_per90")
    goals = _col_mean(att_line, "goals_per90")
    tackles_def = _col_mean(def_line, "tackles_per90")
    tackles_mid = _col_mean(mid_line, "tackles_per90")
    interceptions = _col_mean(def_line, "interceptions_per90")

    if pass_acc >= 86 and key_passes >= 1.2:
        build_up = "中路短传控球"
    elif pass_acc >= 82:
        build_up = "控球推进 / 层层渗透"
    elif key_passes >= 1.5:
        build_up = "前腰串联 / 直塞威胁"
    else:
        build_up = "直接纵向 / 长传冲吊"

    wing_shots = _col_mean(att_line, "shots_per90")
    fullback_attack = _col_mean(def_line, "goals_per90") + _col_mean(def_line, "assists_per90")
    if wing_shots >= 2.5 or fullback_attack >= 0.35:
        width = "边路宽度大 / 翼卫压上"
    elif shots >= 2.0:
        width = "边锋内切 + 肋部渗透"
    else:
        width = "中路堆积 / 窄阵型"

    press = 0.55 * tackles_def + 0.35 * tackles_mid + 0.10 * interceptions
    if press >= 4.5:
        pressing = "高位逼抢 / 前场压迫"
    elif press >= 3.0:
        pressing = "中位逼抢 / 阶段性压迫"
    else:
        pressing = "低位防守 / 保护防线"

    if goals >= 0.45 and shots >= 3.0:
        attack_focus = "多点开花 / 持续施压"
    elif goals >= 0.35:
        attack_focus = "前锋终结 + 二线插上"
    elif key_passes >= 1.0:
        attack_focus = "创造机会为主 / 等待时机"
    else:
        attack_focus = "稳守反击 / 抓转换"

    return {
        "持球构建": build_up,
        "进攻宽度": width,
        "逼抢强度": pressing,
        "进攻重点": attack_focus,
    }


def _style_label(
    attack: float,
    defense: float,
    creativity: float,
    pressing: float,
    tactical: dict[str, str],
) -> str:
    if "控球" in tactical["持球构建"] and creativity >= 5.5:
        return "控球主导 / 技术流"
    if attack >= 7 and creativity >= 6:
        return "进攻主导 / 控球推进"
    if attack >= 6 and pressing >= 5:
        return "高位压迫 / 快速转换"
    if defense >= 6 and attack < 5:
        return "低位防守 / 反击"
    if creativity >= 6:
        return "中场组织 / 控制节奏"
    if attack >= 5:
        return "均衡进攻"
    return "务实防反"


@dataclass
class TeamFootballProfile:
    team: str
    formation_hint: str
    attack_index: float
    creativity_index: float
    defense_index: float
    gk_index: float
    pressing_index: float
    squad_value_top11: float
    avg_age_top11: float
    style_label: str
    position_counts: dict[str, int]
    tactical_breakdown: dict[str, str] = field(default_factory=dict)
    lineup_hint: list[str] = field(default_factory=list)


def build_team_football_profile(merged: pd.DataFrame, team: str) -> TeamFootballProfile | None:
    team = canonical_team_name(team)
    subset = merged[merged["country"] == team].copy()
    if subset.empty:
        return None

    formation, xi = select_starting_xi(subset)
    val_col = _value_col(subset)

    gk_rows = xi[xi["position"].astype(str).map(_pos_group) == "GK"]
    if gk_rows.empty:
        gk_rows = subset[subset["position"].astype(str).str.upper().isin(POS_GK)].nlargest(1, val_col)

    def_line = xi[xi["position"].astype(str).map(_pos_group) == "DEF"]
    mid_line = xi[xi["position"].astype(str).map(_pos_group) == "MID"]
    att_line = xi[xi["position"].astype(str).map(_pos_group) == "ATT"]

    attack_index = (
        0.45 * _col_mean(att_line, "goals_per90")
        + 0.30 * _col_mean(att_line, "shots_per90")
        + 0.25 * _col_mean(att_line, "assists_per90")
    ) * 10.0

    creativity_index = (
        0.40 * _col_mean(mid_line, "key_passes_per90")
        + 0.35 * _col_mean(mid_line, "assists_per90")
        + 0.25 * (_col_mean(mid_line, "pass_accuracy_pct") / 100.0)
    ) * 10.0

    defense_index = (
        0.35 * _col_mean(def_line, "tackles_per90")
        + 0.35 * _col_mean(def_line, "interceptions_per90")
        + 0.30 * _col_mean(def_line, "clearances_per90")
    ) * 3.0

    gk_index = (
        0.60 * _col_mean(gk_rows, "saves_per90")
        + 0.40 * (_col_mean(gk_rows, "rating") / 10.0)
    ) * 5.0

    pressing_index = (
        0.5 * _col_mean(def_line, "tackles_per90")
        + 0.3 * _col_mean(mid_line, "tackles_per90")
        + 0.2 * _col_mean(def_line, "interceptions_per90")
    ) * 4.0

    pos_counts = {
        "GK": int((subset["position"].astype(str).map(_pos_group) == "GK").sum()),
        "DEF": int((subset["position"].astype(str).map(_pos_group) == "DEF").sum()),
        "MID": int((subset["position"].astype(str).map(_pos_group) == "MID").sum()),
        "ATT": int((subset["position"].astype(str).map(_pos_group) == "ATT").sum()),
    }

    tactical = _tactical_breakdown(def_line, mid_line, att_line)
    style = _style_label(attack_index, defense_index, creativity_index, pressing_index, tactical)

    top11_vals = pd.to_numeric(xi[val_col], errors="coerce").fillna(0) if val_col in xi.columns else pd.Series([0])

    def _lineup_label(line: pd.DataFrame) -> str:
        if line.empty or "player_name" not in line.columns:
            return "—"
        names = line["player_name"].astype(str).tolist()
        return " · ".join(names[:6])

    lineup_hint = [
        f"门将: {_lineup_label(gk_rows)}",
        f"后卫: {_lineup_label(def_line)}",
        f"中场: {_lineup_label(mid_line)}",
        f"锋线: {_lineup_label(att_line)}",
    ]

    return TeamFootballProfile(
        team=team,
        formation_hint=formation,
        attack_index=round(attack_index, 2),
        creativity_index=round(creativity_index, 2),
        defense_index=round(defense_index, 2),
        gk_index=round(gk_index, 2),
        pressing_index=round(pressing_index, 2),
        squad_value_top11=float(top11_vals.sum()),
        avg_age_top11=float(pd.to_numeric(xi["age"], errors="coerce").mean()),
        style_label=style,
        position_counts=pos_counts,
        tactical_breakdown=tactical,
        lineup_hint=lineup_hint,
    )


@dataclass
class MatchFootballComparison:
    team_a: TeamFootballProfile
    team_b: TeamFootballProfile
    attack_edge: str
    midfield_edge: str
    defense_edge: str
    gk_edge: str
    overall_edge: str


def compare_football_profiles(pa: TeamFootballProfile, pb: TeamFootballProfile) -> MatchFootballComparison:
    def edge(a: float, b: float, label_a: str, label_b: str) -> str:
        if abs(a - b) < 0.8:
            return "势均力敌"
        return label_a if a > b else label_b

    overall_a = pa.attack_index + pa.creativity_index + pa.defense_index + pa.gk_index
    overall_b = pb.attack_index + pb.creativity_index + pb.defense_index + pb.gk_index

    return MatchFootballComparison(
        team_a=pa,
        team_b=pb,
        attack_edge=edge(pa.attack_index, pb.attack_index, pa.team, pb.team),
        midfield_edge=edge(pa.creativity_index, pb.creativity_index, pa.team, pb.team),
        defense_edge=edge(pa.defense_index, pb.defense_index, pa.team, pb.team),
        gk_edge=edge(pa.gk_index, pb.gk_index, pa.team, pb.team),
        overall_edge=edge(overall_a, overall_b, pa.team, pb.team),
    )


def profile_to_dataframe(profile: TeamFootballProfile) -> pd.DataFrame:
    rows = [
        {"维度": "推测阵型", "数值": profile.formation_hint},
        {"维度": "风格标签", "数值": profile.style_label},
        {"维度": "进攻指数", "数值": f"{profile.attack_index:.1f}"},
        {"维度": "组织/创造力", "数值": f"{profile.creativity_index:.1f}"},
        {"维度": "防守指数", "数值": f"{profile.defense_index:.1f}"},
        {"维度": "门将指数", "数值": f"{profile.gk_index:.1f}"},
        {"维度": "压迫指数", "数值": f"{profile.pressing_index:.1f}"},
        {"维度": "TOP11 RT估值", "数值": f"€{profile.squad_value_top11/1e6:.0f}M"},
    ]
    for key, val in profile.tactical_breakdown.items():
        rows.append({"维度": key, "数值": val})
    return pd.DataFrame(rows)


def tactical_detail_to_dataframe(profile: TeamFootballProfile) -> pd.DataFrame:
    """Line-by-line XI hint for UI."""
    return pd.DataFrame([{"位置": k.split(": ")[0], "预计首发": k.split(": ", 1)[1]} for k in profile.lineup_hint])
