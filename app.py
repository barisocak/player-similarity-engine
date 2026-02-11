import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(layout="wide")

st.markdown("""
<style>
.block-container {
    max-width: 1100px !important;
    margin: auto !important;
    padding-top: 1.5rem !important;
}

div.stButton > button {
    height: 42px !important;
    font-size: 14px !important;
    padding: 0.35rem 0.6rem !important;
}

/* TEXT INPUT CONTAINER */
div[data-testid="stTextInput"] {
    display: flex !important;
    justify-content: center !important;
    margin-top: 20px !important;
    margin-bottom: 35px !important;
}

div[data-testid="stTextInput"] > div {
    width: 520px !important;
}

/* SEARCH INPUT PROPER FIX */
div[data-baseweb="input"] {
    height: 60px !important;
}

/* INPUT STYLING */
div[data-baseweb="input"] input {
    height: 60px !important;
    line-height: 60px !important;
    padding: 0 20px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    box-sizing: border-box !important;
    background: #1c1c1c !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}

/* PLACEHOLDER */
div[data-baseweb="input"] input::placeholder {
    color: rgba(255,255,255,0.45) !important;
}

/* FOCUS EFFECT */
div[data-baseweb="input"] input:focus {
    border: 1px solid #3A86FF !important;
    box-shadow: 0 0 0 2px rgba(58,134,255,0.3) !important;
}

/* ENTER INFO HIDE */
div[data-testid="stTextInput"] small {
    display:none !important;
}

/* SELECTED CARD CLASS */
.player-card {
    background:#141414;
    border-radius:12px;
    padding:14px 18px;
    border:1px solid rgba(255,255,255,0.08);
    margin-bottom:12px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# CONSTANTS
# ======================================================
FILL_OPACITY = 0.55
COLORS = ["#00F5D4","#FF006E","#8338EC","#FB5607","#3A86FF","#FFBE0B"]

POSITION_TABS = ["GK","CB","FB","CM","FW","ST"]

POSITION_TITLE = {
    "GK":"Goalkeeper name...",
    "CB":"Centre Back name...",
    "FB":"Fullback name...",
    "CM":"Midfielder name...",
    "FW":"Forward name...",
    "ST":"Striker name..."
}

DATA_PATH = {
    "GK":"data/GK_dosyasi_ready.xlsx",
    "CB":"data/CB_dosyasi_ready.xlsx",
    "FB":"data/FB_dosyasi_ready.xlsx",
    "CM":"data/CM_dosyasi_ready.xlsx",
    "FW":"data/FW_dosyasi_ready.xlsx",
    "ST":"data/ST_dosyasi_ready.xlsx",
}

RADARS = {
    "GK":{
        "Profil":["GK_Shot_Stopping","GK_Defensive_Org","GK_High_Balls",
                  "GK_Distribution","Save %",
                  "Expected Goals on Target Conceded_p90_INV",
                  "Saves_p90","Pass Accuracy %","Long Pass %"],
        "Stats":["Save %","Expected Goals on Target Conceded_p90_INV",
                 "Saves_p90","Saves (from inside the box)_p90",
                 "Save Percentage (in box)","Pass Accuracy %","Long Pass %"],
        "General":["FM_Duran_Top","FM_Defence","FM_Work_Rate","FM_Physical",
                   "FM_Technique","FM_Intelligence","FM_Goal_Scoring","FM_Devamlilik"]
    },
    "CB":{
        "Profil":["CB_Aerial_Dominance","CB_Ball_Winning","CB_Defence_FM",
                  "CB_Pass_Security","CB_Progressive_Passing",
                  "CB_Risk_Management","CB_Athleticism"],
        "Stats":["Interceptions_p90","Tackles Won_p90","Blocks_p90",
                 "Aerial Duel %","Aerial Duels Won_p90",
                 "Passes, Into Final Third_p90","Pass Accuracy %"],
        "General":["FM_Duran_Top","FM_Defence","FM_Work_Rate","FM_Physical",
                   "FM_Technique","FM_Intelligence","FM_Goal_Scoring","FM_Devamlilik"]
    },
    "FB":{
        "Profil":["Defensive_Contribution","Athleticism_Speed","Ball_Winning",
                  "Progressive_Passing","Cross_Quality","Carrying_Dribbling",
                  "Creative_Impact","Workload"],
        "Stats":["Tackles Won_p90","Interceptions_p90",
                 "Possession Won Final 1/3_p90",
                 "Passes, Into Final Third_p90","Open Play Crosses_p90",
                 "Cross Accuracy %","Dribble Success %"],
        "General":["FM_Duran_Top","FM_Defence","FM_Work_Rate","FM_Physical",
                   "FM_Technique","FM_Intelligence","FM_Goal_Scoring","FM_Devamlilik"]
    },
    "CM":{
        "Profil":["cm_build_up","cm_progressive","cm_creativity",
                  "cm_ball_winning","cm_carrying","cm_defensive_support",
                  "cm_duels","cm_press_resistance",
                  "cm_playmaking","cm_goal_threat"],
        "Stats":["Passes_p90","Pass Accuracy %","Passes, Into Final Third_p90",
                 "Chances Created from Open Play_p90","Expected Assists_p90",
                 "Interceptions_p90","Possession Won_p90",
                 "CM_Creative_Efficiency","CM_Defensive_Balance"],
        "General":["FM_Duran_Top","FM_Defence","FM_Work_Rate","FM_Physical",
                   "FM_Technique","FM_Intelligence","FM_Goal_Scoring","FM_Devamlilik"]
    },
    "FW":{
        "Profil":["Goal_Threat","Finishing","Dribbling",
                  "Progression_Carry","Box_Presence",
                  "Creativity","Passing_Impact","Directness"],
        "Stats":["Expected Goals (ex. Penalties)_p90",
                 "Shots on Target_p90",
                 "Touches in the Opp Box_p90",
                 "Expected Assists_p90",
                 "successfulTakeOn_p90",
                 "Dribble Success %"],
        "General":["FM_Duran_Top","FM_Defence","FM_Work_Rate","FM_Physical",
                   "FM_Technique","FM_Intelligence","FM_Goal_Scoring","FM_Devamlilik"]
    },
    "ST":{
        "Profil":["Goal_Production","Box_Impact","Aerial_Threat",
                  "Off_Ball","Link_Up","Carrying"],
        "Stats":["Goals (ex. Penalties)_p90",
                 "Expected Goals (ex. Penalties)_p90",
                 "Total Shots (inc. Blocks)_p90",
                 "Touches in the Opp Box_p90",
                 "Aerial Duel %","ST_Shot_Accuracy"],
        "General":["FM_Duran_Top","FM_Defence","FM_Work_Rate","FM_Physical",
                   "FM_Technique","FM_Intelligence","FM_Goal_Scoring","FM_Devamlilik"]
    }
}

# ======================================================
# HELPERS
# ======================================================
def norm(x):
    return unidecode(str(x).lower())

def percentile_scale(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(pct=True) * 100

LABEL_OVERRIDE = {
    "FM_Duran_Top": "Set Pieces",
    "FM_Devamlilik": "Consistency"
}

def clean_label(col):
    if col in LABEL_OVERRIDE:
        return LABEL_OVERRIDE[col]
    return col.replace("_"," ").replace("FM ","")

def load_df(pos):
    return pd.read_excel(DATA_PATH[pos])

# ======================================================
# SESSION STATE
# ======================================================
if "position" not in st.session_state:
    st.session_state.position = "GK"

if "picked" not in st.session_state:
    st.session_state.picked = []

# ======================================================
# POSITION TABS
# ======================================================
st.title("Player Similarity Engine")

tab_cols = st.columns(len(POSITION_TABS))
for c, t in zip(tab_cols, POSITION_TABS):
    if c.button(t, use_container_width=True, key=f"tabbtn_{t}"):
        if st.session_state.position != t:
            st.session_state.position = t
            st.session_state.picked = []
        st.rerun()

position = st.session_state.position
df = load_df(position)

# ======================================================
# SEARCH
# ======================================================
st.subheader(POSITION_TITLE[position])
query = st.text_input("", placeholder=POSITION_TITLE[position])

if query:
    q = norm(query)
    matches = df[df["İsim"].apply(norm).str.contains(q)].head(10)

    for _, r in matches.iterrows():
        label = f'{r["İsim"]} ({r["Team"]})'
        if st.button(label, key=f"pick_{position}_{r.name}"):

            exists = any(
                p["pos"] == position and p["name"] == r["İsim"]
                for p in st.session_state.picked
            )

            if not exists:
                st.session_state.picked.append({"pos": position, "name": r["İsim"]})

            st.rerun()

picked_in_pos = [p["name"] for p in st.session_state.picked if p["pos"] == position]
if len(picked_in_pos) == 0:
    st.stop()

# ======================================================
# SELECTED PLAYERS (FIXED)
# ======================================================
st.markdown("### Selected Players")

for name in picked_in_pos:
    row = df[df["İsim"] == name].iloc[0]
    prof = RADARS[position]["Profil"]

    with st.container():
        st.markdown('<div class="player-card">', unsafe_allow_html=True)

        cols = st.columns([0.06, 0.22] + [0.09]*len(prof) + [0.08])

        if cols[0].button("✕", key=f"rm_{position}_{name}"):
            st.session_state.picked = [
                p for p in st.session_state.picked
                if not (p["pos"] == position and p["name"] == name)
            ]
            st.rerun()

        cols[1].markdown(f"""
        <div style='font-weight:700;font-size:16px'>
        {name} <span style='font-weight:400;color:#9aa0a6'>
        ({row['Team']})
        </span>
        </div>
        """, unsafe_allow_html=True)


        for i, metric in enumerate(prof):
            val = row.get(metric, np.nan)
            try:
                vtxt = f"{float(val):.2f}"
            except:
                vtxt = "-"

            cols[2+i].markdown(f"""
            <div style='text-align:center'>
                <div style='font-size:10px;color:#9aa0a6'>{clean_label(metric)}</div>
                <div style='font-size:15px;font-weight:800'>{vtxt}</div>
            </div>
            """, unsafe_allow_html=True)

        cols[-1].markdown(f"""
        <div style='text-align:right'>
            <div style='font-size:10px;color:#9aa0a6'>Min</div>
            <div style='font-size:15px;font-weight:800'>{int(row['Minutes Played'])}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# RADAR
# ======================================================
st.markdown("### Radar Mode")
mode = st.radio("", ["Profil","Stats","General"], horizontal=True)
CURRENT_FEATURES = RADARS[position][mode]

scaled_df = df.copy()
for col in CURRENT_FEATURES:
    scaled_df[col] = percentile_scale(scaled_df[col])

fig = go.Figure()

for i, name in enumerate(picked_in_pos):
    raw = df[df["İsim"] == name][CURRENT_FEATURES].values.flatten()
    scaled = scaled_df[scaled_df["İsim"] == name][CURRENT_FEATURES].values.flatten()
    color = COLORS[i % len(COLORS)]

    fig.add_trace(go.Scatterpolar(
        r=list(scaled) + [scaled[0]],
        theta=[clean_label(x) for x in CURRENT_FEATURES] + [clean_label(CURRENT_FEATURES[0])],
        fill="toself",
        line=dict(color=color, width=4),
        fillcolor=color,
        opacity=FILL_OPACITY,
        name=name,
        customdata=list(raw) + [raw[0]],
        hovertemplate="<b>%{theta}</b><br>Percentile: %{r:.1f}<br>Raw: %{customdata:.2f}<extra></extra>"
    ))

fig.update_layout(
    paper_bgcolor="#0f0f0f",
    plot_bgcolor="#0f0f0f",
    font=dict(color="white"),
    polar=dict(
        bgcolor="#0f0f0f",
        radialaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.25)")
    ),
    height=650
)

st.plotly_chart(fig, use_container_width=True)

# ======================================================
# SIMILAR PLAYERS
# ======================================================
ref = picked_in_pos[-1]
PROFILE_FEATURES = RADARS[position]["Profil"]

X = df[PROFILE_FEATURES].values
ref_vec = df[df["İsim"] == ref][PROFILE_FEATURES].values
sims = cosine_similarity(ref_vec, X)[0]

sim_df = pd.DataFrame({
    "İsim": df["İsim"],
    "Team": df["Team"],
    "Sim": sims
})

sim_df = sim_df[~sim_df["İsim"].isin(picked_in_pos)]
sim_df = sim_df.sort_values("Sim", ascending=False).head(5)

st.markdown(f"### Similar to {ref}")

cols = st.columns(len(sim_df))
for col, (_, r) in zip(cols, sim_df.iterrows()):
    with col:
        st.markdown(f"**{r['İsim']}**  \n{r['Team']}")
        if st.button("Compare", key=f"sim_{position}_{r['İsim']}"):
            st.session_state.picked.append({"pos": position, "name": r["İsim"]})
            st.rerun()
