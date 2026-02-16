import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Kompetenzdiagramm – Pflege-Roboter Team", layout="wide")

# -----------------------------
# Domain-Default (editierbar)
# -----------------------------
DEFAULT_COMPETENCIES = [
    "Empathie",
    "Kommunikation",
    "Systems_Engineering",
    "Safety_Risk",
    "Robotik_HW",
    "Embedded_SW",
    "Perception_AI",
    "Navigation",
    "HRI_UX",
    "Pflegeprozess",
    "Regulatory_Ethik",
    "QA_Testing",
    "Projektmanagement",
]

DEFAULT_DF = pd.DataFrame(
    [
        ["Anna Müller", "HRI/UX", 5, 4, 3, 3, 2, 2, 3, 3, 5, 4, 4, 3, 3],
        ["Ben Schmidt", "Embedded/Safety", 2, 3, 4, 5, 3, 5, 3, 2, 2, 2, 4, 4, 3],
        ["Clara Fischer", "AI/Perception", 2, 3, 3, 3, 2, 3, 5, 3, 3, 2, 3, 3, 2],
        ["David Weber", "Mechatronik", 2, 3, 4, 3, 5, 3, 2, 3, 2, 2, 2, 3, 3],
    ],
    columns=["Name", "Rolle"] + DEFAULT_COMPETENCIES,
)

# -----------------------------
# Helpers
# -----------------------------
NON_SKILL_COLS = {"Name", "Rolle", "Kommentar"}

def _try_read_csv(text: str) -> pd.DataFrame:
    # robust: try ; then , then auto with python engine
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            if "Name" in df.columns and len(df.columns) >= 2:
                return df
        except Exception:
            pass
    # fallback
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

def _sanitize_colname(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-zÄÖÜäöüß_]+", "", s)
    return s

def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def radar_figure(df: pd.DataFrame, competencies: list[str], members: list[str], scale_max: float) -> go.Figure:
    fig = go.Figure()
    if not competencies:
        return fig

    # closed loop
    categories = competencies + [competencies[0]]

    for name in members:
        row = df[df["Name"] == name]
        if row.empty:
            continue
        vals = row.iloc[0][competencies].astype(float).tolist()
        vals = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=categories, fill="toself", name=name))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, scale_max])
        ),
        showlegend=True,
        margin=dict(l=30, r=30, t=30, b=30),
    )
    return fig

def team_aggregate(df: pd.DataFrame, competencies: list[str], members: list[str], method: str) -> pd.Series:
    sub = df[df["Name"].isin(members)][competencies].astype(float)
    if sub.empty:
        return pd.Series(index=competencies, dtype=float)

    if method == "Mittelwert":
        return sub.mean(axis=0, skipna=True)
    if method == "Maximum":
        return sub.max(axis=0, skipna=True)
    if method == "Minimum":
        return sub.min(axis=0, skipna=True)
    # default
    return sub.mean(axis=0, skipna=True)

# -----------------------------
# UI
# -----------------------------
st.title("Kompetenzdiagramm für die Neuentwicklung eines Altenpflege-Roboters")

with st.sidebar:
    st.header("Datenquelle")
    uploaded = st.file_uploader("CSV hochladen", type=["csv", "clv", "txt"])

    st.caption("Hinweis: Trennzeichen `;` wird bevorzugt, `,` wird ebenfalls erkannt.")

    st.header("Skalierung")
    scale_max = st.number_input("Maximalwert (Skala)", min_value=1.0, max_value=100.0, value=5.0, step=1.0)

    st.header("Ansichten")
    show_heatmap = st.checkbox("Kompetenz-Matrix (Heatmap) anzeigen", value=True)
    show_radar = st.checkbox("Radar-Diagramm anzeigen", value=True)
    show_team_radar = st.checkbox("Team-Aggregation (Radar) anzeigen", value=True)

# Load data
if uploaded is not None:
    text = uploaded.getvalue().decode("utf-8", errors="replace")
    try:
        df = _try_read_csv(text)
    except Exception as e:
        st.error(f"CSV konnte nicht gelesen werden: {e}")
        df = DEFAULT_DF.copy()
else:
    df = DEFAULT_DF.copy()

# Sanitize column names (optional but helps)
df = df.rename(columns={c: _sanitize_colname(c) for c in df.columns})

# Ensure required column
if "Name" not in df.columns:
    st.error("CSV muss eine Spalte 'Name' enthalten.")
    st.stop()

# Identify competency columns
competency_cols = [c for c in df.columns if c not in NON_SKILL_COLS]

# Coerce numeric on competencies
df = _coerce_numeric(df, competency_cols)

# Controls: add/remove competencies & members
c1, c2, c3 = st.columns([1.2, 1.2, 1.6])

with c1:
    st.subheader("Kompetenzen auswählen")
    competencies = st.multiselect(
        "Welche Kompetenzen sollen im Diagramm erscheinen?",
        options=competency_cols,
        default=competency_cols[: min(len(competency_cols), 10)],
    )

    new_comp = st.text_input("Neue Kompetenz hinzufügen (Spaltenname)")
    if st.button("Kompetenz anlegen"):
        if new_comp:
            new_comp = _sanitize_colname(new_comp)
            if new_comp in df.columns:
                st.warning("Kompetenz existiert bereits.")
            else:
                df[new_comp] = 0.0
                competencies = list(dict.fromkeys(competencies + [new_comp]))
                st.success(f"Kompetenz '{new_comp}' hinzugefügt.")

with c2:
    st.subheader("Teammitglieder")
    members_all = df["Name"].astype(str).tolist()
    members = st.multiselect("Mitglieder im Radar", options=members_all, default=members_all[: min(len(members_all), 4)])

    new_member = st.text_input("Neues Mitglied hinzufügen (Name)")
    new_role = st.text_input("Rolle (optional)")
    if st.button("Mitglied anlegen"):
        if new_member:
            if (df["Name"].astype(str) == str(new_member)).any():
                st.warning("Mitglied existiert bereits.")
            else:
                row = {c: np.nan for c in df.columns}
                row["Name"] = str(new_member)
                if "Rolle" in df.columns:
                    row["Rolle"] = str(new_role) if new_role else ""
                for c in competency_cols:
                    row[c] = 0.0
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                st.success(f"Mitglied '{new_member}' hinzugefügt.")

    drop_members = st.multiselect("Mitglieder löschen", options=members_all, default=[])
    if st.button("Ausgewählte löschen"):
        if drop_members:
            df = df[~df["Name"].isin(drop_members)].reset_index(drop=True)
            st.success("Mitglieder gelöscht.")

with c3:
    st.subheader("Daten direkt bearbeiten")
    st.caption("Tabelle ist interaktiv editierbar. Werte außerhalb der Skala sind erlaubt, aber die Radar-Achse nutzt den Maximalwert aus der Sidebar.")
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    df = edited.copy()

# Refresh competencies after possible edits
competency_cols = [c for c in df.columns if c not in NON_SKILL_COLS]
if not competencies:
    competencies = competency_cols

# Charts
left, right = st.columns([1, 1])

with left:
    if show_radar:
        st.subheader("Radar-Kompetenzdiagramm (pro Mitglied)")
        fig = radar_figure(df, competencies, members, scale_max=scale_max)
        st.plotly_chart(fig, use_container_width=True)

    if show_team_radar:
        st.subheader("Radar-Kompetenzdiagramm (Team-Aggregation)")
        method = st.selectbox("Aggregation", ["Mittelwert", "Maximum", "Minimum"], index=0)
        agg = team_aggregate(df, competencies, members, method=method)

        if not agg.empty:
            categories = competencies + [competencies[0]]
            vals = agg.tolist()
            vals = vals + [vals[0]]

            fig_team = go.Figure()
            fig_team.add_trace(go.Scatterpolar(r=vals, theta=categories, fill="toself", name=f"Team ({method})"))
            fig_team.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, scale_max])),
                showlegend=True,
                margin=dict(l=30, r=30, t=30, b=30),
            )
            st.plotly_chart(fig_team, use_container_width=True)
        else:
            st.info("Keine Daten für Team-Aggregation (Mitglieder/Kompetenzen prüfen).")

with right:
    if show_heatmap:
        st.subheader("Kompetenz-Matrix (Heatmap)")
        # Build matrix: rows=members, cols=competencies
        df_hm = df[["Name"] + competencies].copy()
        df_hm = _coerce_numeric(df_hm, competencies)
        df_hm = df_hm.set_index("Name")
        fig_hm = px.imshow(df_hm, aspect="auto")
        fig_hm.update_layout(margin=dict(l=30, r=30, t=30, b=30))
        st.plotly_chart(fig_hm, use_container_width=True)

# Export
st.divider()
st.subheader("Export")

csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False, sep=";")
st.download_button(
    "Aktualisierte CSV herunterladen",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name="team_kompetenzen_aktualisiert.csv",
    mime="text/csv",
)

# Optional: Export charts as HTML
if show_radar:
    html = fig.to_html(include_plotlyjs="cdn")
    st.download_button(
        "Radar als HTML exportieren",
        data=html.encode("utf-8"),
        file_name="kompetenz_radar.html",
        mime="text/html",
    )
