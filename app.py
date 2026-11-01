import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Churn Radar", page_icon="📡", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "churn_model.pkl")
DATA_PATH = os.path.join(BASE, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

INK, TEAL, CORAL, AMBER, SAGE, MIST = (
    "#0F2A3D", "#14919B", "#E4572E", "#F2A541", "#3FA877", "#F3F7F8")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=Manrope:wght@400;500;600&display=swap');
html, body, .stApp {{ font-family: 'Manrope', sans-serif; }}
.stApp {{ background: {MIST}; }}
.block-container {{ padding-top: 2rem; max-width: 1150px; }}
h1, h2, h3, h4 {{ font-family: 'Sora', sans-serif; color: {INK}; }}

.hero {{
  background: linear-gradient(120deg, {INK} 0%, #1B4A5E 55%, {TEAL} 100%);
  border-radius: 20px; padding: 2.2rem 2.4rem; margin-bottom: 1.2rem;
}}
.hero-title {{ font-family: 'Sora', sans-serif; font-weight: 700;
  font-size: 2.4rem; color: #fff; line-height: 1.15; }}
.hero-sub {{ color: #D6E8EC; margin-top: .6rem; font-size: 1.05rem; max-width: 620px; }}

[data-testid="stForm"] {{
  background: #fff; border: 1px solid #E1EAEC; border-radius: 18px; padding: 1.4rem;
}}
[data-testid="stFormSubmitButton"] button {{
  background: {TEAL}; color: #fff; border: none; border-radius: 10px;
  width: 100%; padding: .65rem 1rem; font-weight: 600;
}}
[data-testid="stFormSubmitButton"] button:hover {{ background: {INK}; color: #fff; }}

.card {{ background: #fff; border: 1px solid #E1EAEC; border-radius: 18px; padding: 1.4rem 1.6rem; }}
.kpi {{ background: #fff; border: 1px solid #E1EAEC; border-radius: 14px; padding: 1rem 1.2rem; }}
.kpi-v {{ font-family: 'Sora', sans-serif; font-size: 1.8rem; font-weight: 700; color: {INK}; }}
.kpi-l {{ color: #5B7280; font-size: .9rem; }}
.pill {{ display: inline-block; padding: .3rem .9rem; border-radius: 999px;
  color: #fff; font-weight: 600; font-size: .95rem; }}
.action {{ border-left: 4px solid {TEAL}; background: {MIST}; border-radius: 8px;
  padding: .6rem .9rem; margin-top: .55rem; color: {INK}; }}
.muted {{ color: #5B7280; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                                labels=["0-12", "13-24", "25-48", "49-72"])
    return df


try:
    model = load_model()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load churn_model.pkl: {e}")
    st.stop()

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
ADDONS = ["Online Security", "Online Backup", "Device Protection",
          "Tech Support", "Streaming TV", "Streaming Movies"]


def build_row(tenure, contract, internet, payment, monthly, addons, extra):
    yn = lambda flag: "Yes" if flag else "No"  # noqa: E731
    has = lambda name: internet != "No" and name in addons  # noqa: E731

    total = float(monthly * tenure)
    avg_monthly = total / tenure if tenure > 0 else float(monthly)
    group = ("0-12" if tenure <= 12 else "13-24" if tenure <= 24
             else "25-48" if tenure <= 48 else "49-72")

    return pd.DataFrame([{
        "gender": extra["gender"],
        "SeniorCitizen": int(extra["senior"]),
        "Partner": yn(extra["partner"]),
        "Dependents": yn(extra["dependents"]),
        "tenure": tenure,
        "PhoneService": yn(extra["phone"]),
        "MultipleLines": yn(extra["phone"] and extra["multi"]),
        "InternetService": internet,
        "OnlineSecurity": yn(has("Online Security")),
        "OnlineBackup": yn(has("Online Backup")),
        "DeviceProtection": yn(has("Device Protection")),
        "TechSupport": yn(has("Tech Support")),
        "StreamingTV": yn(has("Streaming TV")),
        "StreamingMovies": yn(has("Streaming Movies")),
        "Contract": contract,
        "PaperlessBilling": yn(extra["paperless"]),
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": total,
        "tenure_group": group,
        "num_services": sum(has(a) for a in ADDONS),
        "avg_monthly": avg_monthly,
        "charge_diff": float(monthly) - avg_monthly,
    }])


def risk_level(p):
    if p >= 0.60:
        return "High risk", CORAL
    if p >= 0.35:
        return "Medium risk", AMBER
    return "Low risk", SAGE


def recommendations(p, contract, internet, payment, addons):
    tips = []
    if p < 0.35:
        return ["Keep the experience steady. A loyalty perk or a thank-you note works well here."]
    if contract == "Month-to-month":
        tips.append("Offer a discount for moving to a 1-year or 2-year contract.")
    if payment == "Electronic check":
        tips.append("Encourage automatic payments with a small bill credit.")
    if internet == "Fiber optic" and "Tech Support" not in addons:
        tips.append("Add a free 3-month Tech Support trial to the fiber plan.")
    if not tips:
        tips.append("Schedule a proactive check-in call to understand any issues.")
    if p >= 0.60:
        tips.append("Prioritise this customer in this week's retention outreach.")
    return tips[:3]


def gauge(p):
    _, color = risk_level(p)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": "%", "font": {"size": 46, "family": "Sora", "color": INK}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9FB2BB"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#EEF3F4", "borderwidth": 0,
            "steps": [{"range": [0, 35], "color": "#E3F3EA"},
                      {"range": [35, 60], "color": "#FBEFD9"},
                      {"range": [60, 100], "color": "#F9E0D9"}],
        }))
    fig.update_layout(height=240, margin=dict(l=25, r=25, t=15, b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def style(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Manrope", color=INK),
                      title=dict(font=dict(family="Sora", size=16)),
                      showlegend=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E6EEF0")
    return fig


def rate_bar(df, col, title, order=None, horizontal=False):
    g = df.groupby(col, observed=True)["Churn"].mean().mul(100).round(1).reset_index()
    if order:
        g[col] = pd.Categorical(g[col], order, ordered=True)
        g = g.sort_values(col)
    colors = [CORAL if v == g["Churn"].max() else TEAL for v in g["Churn"]]
    labels = [f"{v}%" for v in g["Churn"]]
    cats = g[col].astype(str)
    if horizontal:
        fig = go.Figure(go.Bar(y=cats, x=g["Churn"], orientation="h",
                               marker_color=colors, text=labels, textposition="outside"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, g["Churn"].max() * 1.2])
    else:
        fig = go.Figure(go.Bar(x=cats, y=g["Churn"], marker_color=colors,
                               text=labels, textposition="outside"))
        fig.update_yaxes(range=[0, g["Churn"].max() * 1.25])
    fig.update_layout(title=title)
    return style(fig)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-title">Churn Radar</div>
  <div class="hero-sub">Find out which telecom customers are likely to leave,
  and what to offer them before they do.</div>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_insights = st.tabs(["Predict churn", "Insights and model"])

# ----------------------------------------------------------------------------
# Tab 1: Prediction
# ----------------------------------------------------------------------------
with tab_predict:
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        with st.form("customer"):
            st.markdown("#### Customer details")
            tenure = st.slider("Months as a customer", 0, 72, 12)
            contract = st.radio("Contract", ["Month-to-month", "One year", "Two year"],
                                horizontal=True)
            c1, c2 = st.columns(2)
            internet = c1.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            payment = c2.selectbox("Payment method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.slider("Monthly bill ($)", 18, 120, 70)

            with st.expander("More details (optional)"):
                addons = st.multiselect("Add-on services they have", ADDONS)
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior = st.checkbox("Senior citizen")
                partner = st.checkbox("Has a partner")
                dependents = st.checkbox("Has dependents")
                phone = st.checkbox("Has phone service", value=True)
                multi = st.checkbox("Multiple phone lines")
                paperless = st.checkbox("Paperless billing", value=True)

            submitted = st.form_submit_button("Check churn risk")

    with right:
        if not submitted:
            st.markdown("""
            <div class="card">
              <h4 style="margin-top:0">Your prediction will appear here</h4>
              <p class="muted">Fill in the customer's details and press
              <b>Check churn risk</b>. Contract, tenure and monthly bill have the
              biggest effect, so those are enough for a good estimate.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            extra = dict(gender=gender, senior=senior, partner=partner,
                         dependents=dependents, phone=phone, multi=multi,
                         paperless=paperless)
            row = build_row(tenure, contract, internet, payment, monthly, addons, extra)
            prob = float(model.predict_proba(row)[0, 1])
            label, color = risk_level(prob)

            st.markdown(f"""
            <div class="card">
              <span class="pill" style="background:{color}">{label}</span>
              <span class="muted" style="margin-left:.6rem">chance this customer leaves</span>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(gauge(prob))

            tips = recommendations(prob, contract, internet, payment, addons)
            tips_html = "".join(f'<div class="action">{t}</div>' for t in tips)
            st.markdown(f"""
            <div class="card">
              <h4 style="margin-top:0">Suggested next steps</h4>{tips_html}
            </div>
            """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Tab 2: Insights and model
# ----------------------------------------------------------------------------
with tab_insights:
    if not os.path.exists(DATA_PATH):
        st.warning("Put WA_Fn-UseC_-Telco-Customer-Churn.csv in the same folder as "
                   "app.py to see the charts.")
    else:
        df = load_data()

        k1, k2, k3, k4 = st.columns(4)
        kpis = [
            (f"{len(df):,}", "Customers analysed"),
            (f"{df['Churn'].mean() * 100:.1f}%", "Overall churn rate"),
            (f"{df['tenure'].mean():.0f} months", "Average tenure"),
            (f"${df['MonthlyCharges'].mean():.0f}", "Average monthly bill"),
        ]
        for col, (v, l) in zip((k1, k2, k3, k4), kpis):
            col.markdown(f'<div class="kpi"><div class="kpi-v">{v}</div>'
                         f'<div class="kpi-l">{l}</div></div>', unsafe_allow_html=True)

        by_contract = df.groupby("Contract")["Churn"].mean() * 100
        by_tenure = df.groupby("tenure_group", observed=True)["Churn"].mean() * 100
        st.markdown(f"""
        <div class="card" style="margin-top:1rem">
          <h4 style="margin-top:0">What the data says</h4>
          <div class="action">Month-to-month customers churn at
          <b>{by_contract['Month-to-month']:.0f}%</b>, compared with
          <b>{by_contract['Two year']:.0f}%</b> on two-year contracts.</div>
          <div class="action">Customers in their first year churn at
          <b>{by_tenure['0-12']:.0f}%</b>, falling to
          <b>{by_tenure['49-72']:.0f}%</b> after four years.</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        a, b = st.columns(2)
        a.plotly_chart(rate_bar(df, "Contract", "Churn rate by contract",
                                ["Month-to-month", "One year", "Two year"]))
        b.plotly_chart(rate_bar(df, "tenure_group", "Churn rate by tenure (months)",
                                ["0-12", "13-24", "25-48", "49-72"]))
        c, d = st.columns(2)
        c.plotly_chart(rate_bar(df, "InternetService", "Churn rate by internet service"))
        d.plotly_chart(rate_bar(df, "PaymentMethod", "Churn rate by payment method",
                                horizontal=True))

    st.markdown("### Model")
    m1, m2 = st.columns([1.2, 1], gap="large")

    with m1:
        try:
            names = model.named_steps["prep"].get_feature_names_out()
            imp = model.named_steps["model"].feature_importances_
            top = (pd.DataFrame({"f": [n.split("__", 1)[-1].replace("_", " ") for n in names],
                                 "v": imp}).nlargest(12, "v").iloc[::-1])
            fig = go.Figure(go.Bar(x=top["v"], y=top["f"], orientation="h",
                                   marker_color=TEAL))
            fig.update_layout(title="Top features driving predictions")
            st.plotly_chart(style(fig, 420))
        except Exception:  # noqa: BLE001
            st.info("Feature importances are not available for this model.")

    with m2:
        st.markdown("**Results on the held-out test set (20% of customers)**")
        results = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
            "Precision": [0.498, 0.564, 0.516],
            "Recall": [0.791, 0.650, 0.781],
            "F1": [0.612, 0.604, 0.621],
            "ROC-AUC": [0.842, 0.827, 0.837],
        })
        st.dataframe(results, hide_index=True)
        st.markdown(
            '<p class="muted">Recall matters most for churn: missing a customer who '
            'leaves costs far more than sending one extra retention offer.</p>',
            unsafe_allow_html=True)