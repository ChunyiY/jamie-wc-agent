<p align="center">
  <img src="assets/jamie-vardy-fm24-golden-boot.png" alt="Jamie Vardy — FM24 Championship Golden Boot, Norwich City" width="720"/>
</p>

<h1 align="center">Jamie: The only WC agent you need</h1>
<h3 align="center">Jamie · 你唯一需要的世界杯智能体</h3>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/tests-60%20passing-brightgreen.svg" alt="Tests"/>
  <img src="https://img.shields.io/badge/FM24-诺维奇冲超-yellow.svg" alt="FM24 Norwich"/>
  <img src="https://img.shields.io/badge/瓦尔迪-30球-green.svg" alt="Vardy 30 goals"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="MIT"/>
</p>

---

## 🏆 In memory of a legend / 纪念一位传奇

**EN:** This repo exists because I love football — the real kind, the spreadsheet kind, and the *"one more season"* kind. In **Football Manager 2024**, I took **Norwich City** up the Championship ladder with a 37-year-old Jamie Vardy who refused to act his age: **30 league goals**, **Golden Boot**, second in the table, promotion push alive. The man literally sprinted like the save file was on 3× speed.

**中文：** 这个项目的存在，是因为我实在太爱足球了——真球场上的、数据表里的、以及「再开一轮存档」那种。在 **FM24** 里，我执教 **诺维奇城**，阵中站着一位拒绝服老的 **37 岁杰米·瓦尔迪**：单赛季 **30 球**、**英冠金靴**、积分榜紧咬升级区。老爷子跑起来，仿佛我的 FM 开了三倍速。

**EN:** So when I built a World Cup 2026 analysis agent, of course it had to be called **Jamie**. Not because it predicts Vardy will start for England in 2026 *(…probably)* — but because every good football project deserves a poacher who shows up when it matters.

**中文：** 所以当我做这套 **2026 世界杯** 分析智能体时，它必须叫 **Jamie**。不是因为模型预测瓦尔迪会在 2026 世界杯首发 *(……大概不会)* —— 而是因为每个像样的足球项目，都值得拥有一位「关键场次必定在线」的偷猎型前锋。

> *"He's 37 on the calendar and 23 in the channel."*  
> *「户口本写 37，肋部冲刺像 23。」*  
> — 主教练 **杨淳亦**, FM24 赛后采访（大概）

---

## ⚽ What Jamie actually does / Jamie 到底是干嘛的

**EN:** Jamie is a local **Streamlit** dashboard for **FIFA World Cup 2026** — ML win/draw/loss probabilities, scorelines, xG, tactics, squad intel, and conservative paper-trading notes. Built for fans and researchers. **Not** a money printer. Vardy would still miss a one-on-one in the 93rd minute; this model will miss too.

**中文：** Jamie 是一个本地 **Streamlit** 世界杯助手：**2026 美加墨** 赛程、集成 ML 胜平负概率、精确比分、xG、战术面板、球员情报，以及偏保守的模拟下注参考。给球迷和研究者用。**不能**保证盈利。瓦尔迪单刀不进，模型也会翻车——这很足球。

| Feature / 功能 | EN | 中文 |
|----------------|----|------|
| Schedule | WC 2026 fixtures & groups | 世界杯赛程与小组 |
| ML stack | Dixon–Coles + ordered logit + multinomial | 多模型集成 + 滚动验证 |
| Scores | Exact scores, O/U 2.5, BTTS | 比分矩阵、大小球、双方进球 |
| Tactics | Formation hints, xG/SOT, mentality | 阵型推测、比赛质量、心态指标 |
| Players | Squads, per-90, H2H scorers | 名单、俱乐部数据、历史交锋射手 |
| Markets | Paper-trading vs Polymarket (optional) | 纸面交易参考（默认不开实盘） |

> ⚠️ **Disclaimer / 免责声明**  
> EN: Not financial or gambling advice. Model output ≠ truth.  
> 中文：非财务或博彩建议。模型输出不等于赛果。请理性看球、量力而行。

---

## 🚀 Quick start / 快速开始

```bash
git clone https://github.com/ChunyiY/jamie-wc-agent.git
cd jamie-wc-agent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
python scripts/download_data.py --all
streamlit run app.py
```

**EN:** Open **http://localhost:8501** → pick a match → **Start ML analysis**.  
First run trains the model (~30–90s). Hit **Full retrain + calibration** in the sidebar after updating data.

**中文：** 打开 **http://localhost:8501** → 选一场比赛 → **开始 ML 分析**。  
首次运行会训练模型（约 30–90 秒）。更新数据后，在侧边栏点 **完整重训 + 校准**。

---

## 📁 Project layout / 项目结构

```
jamie-wc-agent/
├── app.py                  # Streamlit UI / 前端
├── assets/                 # Jamie Vardy FM24 shrine / 瓦尔迪纪念照
├── src/                    # Core library / 核心逻辑
│   ├── match_analyzer.py   # Match prediction pipeline
│   ├── model_loader.py     # Train & cache ensemble
│   ├── football_profile.py # Formation & tactical profile
│   └── ...
├── data/                   # CSV datasets (see data/README.md)
├── models/                 # Trained weights (.joblib, gitignored)
├── tests/                  # pytest — 60 tests
└── scripts/download_data.py
```

---

## 📊 Data sources / 数据来源

| Dataset | Provider |
|---------|----------|
| International results & scorers | [martj42/international_results](https://github.com/martj42/international_results) |
| WC 2026 fixtures | [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json) |
| Squads & per-90 | [risingtransfers/world-cup-2026-data](https://github.com/risingtransfers/world-cup-2026-data) |
| xG / match stats | [StatsBomb Open Data](https://github.com/statsbomb/open-data) |
| WC2026 enriched | [mominullptr/FIFA-World-Cup-2026-Dataset](https://github.com/mominullptr/FIFA-World-Cup-2026-Dataset) |

---

## ⚙️ Configuration / 配置

Copy `.env.example` → `.env`

| Variable | Default | EN | 中文 |
|----------|---------|----|------|
| `DEFAULT_BANKROLL` | `70` | Paper-trade bankroll ($) | 模拟资金池 |
| `USER_COUNTRY_CODE` | `US` | Compliance routing | 合规地区 |
| `POLYMARKET_PLATFORM` | `us` | `us` or `international` | 市场平台 |
| `ENABLE_LIVE_TRADING` | `false` | Keep false | 请勿开启实盘 |

---

## 🧪 Tests / 测试

```bash
pytest tests/ -q
```

CI runs on push → `.github/workflows/ci.yml`

---

## 💰 Small bankroll mode (~$70) / 小资金模式

**EN:** Defaults tuned for a patient ~$70 bankroll — max 0.5% per trade, 6% total exposure, 7% min edge. Most markets → **NO_TRADE**. (Vardy would approve of the patience; he'd just never approve of passing instead of shooting.)

**中文：** 默认按约 **$70** 小资金设计——单笔风险 0.5%、总敞口 6%、最低边际 7%。绝大多数场次会显示 **NO_TRADE**。（瓦尔迪认可耐心；但他永远不会认可「有机会不传射」。）

---

## 📜 License / 许可证

MIT — use responsibly and only where permitted by law.  
MIT — 请合法、理性使用。

---

<p align="center">
  <b>🟡🟢 Once Canaries, always Canaries. / 金丝雀永飞。</b><br/>
  <sub>FM24 save archived. Jamie lives on in Python. / FM 存档会封盘，Jamie 在代码里继续跑。</sub>
</p>
