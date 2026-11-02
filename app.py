import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# 1. PAGE SETTINGS
# =========================================================
st.set_page_config(
    page_title="Credit Portfolio Segmentation",
    page_icon="📊",
    layout="wide"
)

BASE = Path(__file__).parent
DATA = BASE / "data" / "processed"

# =========================================================
# 2. GREEN DESIGN AND CUSTOM STYLING
# =========================================================
st.markdown("""
<style>
.stApp {
    background: #F3FAF4;
    color: #203629;
}

h1, h2, h3, p, label, .stMarkdown {
    color: #203629;
}

h1 {
    color: #176534;
    font-weight: 800;
}

h2, h3 {
    color: #237442;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #000000;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: #191919;
    border-radius: 9px;
    padding: 8px;
    margin-bottom: 5px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #287D46;
}

[data-testid="stSidebar"] hr {
    border-color: #444444;
}

/* KPI CARDS */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #C7E6CE;
    border-radius: 14px;
    padding: 15px;
    box-shadow: 0 3px 12px #174D3010;
    transition: transform .2s ease, box-shadow .2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px #174D3020;
}

[data-testid="stMetricLabel"] {
    color: #42634D !important;
}

[data-testid="stMetricValue"] {
    color: #176534 !important;
}

/* INPUT FIELDS */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: #FFFFFF !important;
    color: #203629 !important;
    border-radius: 10px !important;
}

[data-testid="stNumberInput"] input {
    border: 1px solid #A8D4B2 !important;
}

/* DROPDOWN DESIGN */
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    border: 1.5px solid #A8D4B2 !important;
    min-height: 45px;
    transition: all .2s ease;
}

[data-testid="stSelectbox"] div[data-baseweb="select"]:hover {
    border-color: #258448 !important;
    box-shadow: 0 0 0 3px #54B87525;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: #203629 !important;
    font-weight: 600;
}

/* DROPDOWN OPTIONS */
div[data-baseweb="popover"] ul {
    background: #FFFFFF !important;
    border: 1px solid #C7E6CE;
    border-radius: 10px;
    padding: 5px;
}

div[data-baseweb="menu"] li:hover {
    background: #E3F5E8 !important;
    color: #176534 !important;
}

div[data-baseweb="menu"] li[aria-selected="true"] {
    background: #D5F0DC !important;
    color: #176534 !important;
    font-weight: 700;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(90deg, #8ED9A1, #54B875);
    color: #123D22 !important;
    border: none;
    border-radius: 11px;
    padding: 10px 22px;
    font-weight: 700;
    transition: all .2s ease;
}

.stButton > button:hover {
    background: #258448;
    color: #FFFFFF !important;
    transform: translateY(-2px);
    box-shadow: 0 5px 14px #25844840;
}

/* TABS */
[data-testid="stTabs"] button {
    color: #237442;
    font-weight: 700;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #176534 !important;
    border-bottom-color: #258448 !important;
}

/* DIVIDERS */
hr {
    border-color: #C7E6CE;
}

/* ANIMATION */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(7px); }
    to { opacity: 1; transform: translateY(0); }
}

[data-testid="stVerticalBlock"] {
    animation: fadeIn .35s ease;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. LOAD DATA AND MODEL
# =========================================================
@st.cache_data
def load_csv(name, **kwargs):
    return pd.read_csv(DATA / name, **kwargs)


@st.cache_resource
def load_model():
    with open(BASE / "models" / "model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()
summary = load_csv("app_cluster_summary.csv")
sample = load_csv("dashboard_sample.csv")
by_year = load_csv("app_default_by_year.csv", index_col=0)
algos = load_csv("app_algorithm_comparison.csv")

names = list(summary["segment"])
colors = dict(zip(names, px.colors.qualitative.Bold))
sample["cluster_name"] = pd.Categorical(
    sample["cluster_name"], names, ordered=True
)

overall_rate = (
    summary["defaults"].sum() / summary["borrowers"].sum() * 100
)

# Different graph colour palettes
PALETTE_1 = px.colors.qualitative.Bold
PALETTE_2 = px.colors.qualitative.Vivid
PALETTE_3 = px.colors.qualitative.Safe
PALETTE_4 = px.colors.qualitative.Pastel

# =========================================================
# 4. TITLE AND NAVIGATION
# =========================================================
st.title("📊 Credit Portfolio Segmentation")
st.caption(
    "Discovering borrower groups using unsupervised machine learning. "
    "Default information was excluded during clustering and used only "
    "to analyse repayment patterns."
)

tab_overview, tab_segments, tab_default, tab_map, tab_try, tab_insights = st.tabs(
    ["📌 Overview", "👥 Segments", "📉 Default Rates",
     "🗺️ Cluster Map", "🧑‍💼 Try a Borrower", "💡 Insights"]
)

# =========================================================
# 5. OVERVIEW
# =========================================================
with tab_overview:
    kmeans_row = algos[algos["algorithm"] == "K-Means"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Borrowers", f"{int(summary['borrowers'].sum()):,}")
    c2.metric("Overall Default", f"{overall_rate:.1f}%")
    c3.metric("Total Segments", len(summary))
    c4.metric("Silhouette Score", f"{kmeans_row['silhouette']:.3f}")

    left, right = st.columns(2)

    with left:
        fig = px.pie(
            summary, values="borrowers", names="segment",
            hole=0.5, color="segment", color_discrete_map=colors,
            title="Borrower Distribution by Segment"
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, width="stretch")

    with right:
        fig = px.bar(
            summary, x="segment", y="default_rate_pct",
            color="segment",
            color_discrete_map=dict(zip(names, PALETTE_2)),
            text=summary["default_rate_pct"].round(1),
            title="Default Rate Across Segments"
        )
        fig.add_hline(
            y=overall_rate, line_dash="dash",
            line_color="#111827",
            annotation_text=f"Average {overall_rate:.1f}%"
        )
        fig.update_layout(
            showlegend=False, xaxis_title="",
            yaxis_title="Default Rate (%)"
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Algorithm Performance Comparison")
    st.dataframe(
        algos[["algorithm", "n_clusters", "noise_pct", "silhouette",
               "davies_bouldin", "ARI_vs_kmeans"]],
        width="stretch", hide_index=True
    )
    st.caption(
        "Higher Silhouette and lower Davies-Bouldin scores generally "
        "indicate better-separated clusters. K-Means was selected for "
        "its practical and interpretable segments."
    )

# =========================================================
# 6. SEGMENTS
# =========================================================
with tab_segments:
    chosen = st.selectbox("Select a borrower segment", names)
    row = summary[summary["segment"] == chosen].iloc[0]

    c = st.columns(5)
    c[0].metric("Borrowers", f"{int(row['borrowers']):,}",
                f"{row['share_pct']:.1f}% of portfolio")
    c[1].metric("Median Income", f"${row['median_income']:,.0f}")
    c[2].metric("Median FICO", f"{row['median_fico']:.0f}")
    c[3].metric("Median Loan", f"${row['median_loan_amnt']:,.0f}")
    c[4].metric("Default Rate", f"{row['default_rate_pct']:.1f}%")

    c = st.columns(5)
    c[0].metric("Median DTI", f"{row['median_dti']:.1f}")
    c[1].metric("Avg. Employment", f"{row['mean_emp_years']:.1f} years")
    c[2].metric("Loan / Income", f"{row['median_loan_to_income']:.2f}")
    c[3].metric("Renters", f"{row['pct_rent']:.0f}%")
    c[4].metric("Debt Consolidation",
                f"{row['pct_debt_consolidation']:.0f}%")

    st.subheader("Compare Borrower Segments")
    metric = st.selectbox(
        "Choose a comparison measure",
        [col for col in summary.columns
         if col not in ("segment", "borrowers", "defaults")],
        index=3
    )

    fig = px.bar(
        summary, x="segment", y=metric, color="segment",
        color_discrete_sequence=PALETTE_3
    )
    fig.update_layout(showlegend=False, xaxis_title="")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Feature Distribution Within Each Segment")
    feature = st.selectbox(
        "Choose a borrower feature",
        ["fico_n", "revenue", "dti_n", "loan_amnt", "emp_years"]
    )

    data = sample
    if feature == "revenue":
        data = sample[
            sample["revenue"] <= sample["revenue"].quantile(0.99)
        ]

    fig = px.box(
        data, x="cluster_name", y=feature, color="cluster_name",
        color_discrete_map=dict(zip(names, PALETTE_4)),
        category_orders={"cluster_name": names}
    )
    fig.update_layout(showlegend=False, xaxis_title="")
    st.plotly_chart(fig, width="stretch")

# =========================================================
# 7. DEFAULT RATES
# =========================================================
with tab_default:
    st.info(
        "Default was excluded from clustering. It is analysed here "
        "to compare repayment behaviour across the borrower groups."
    )

    fig = px.bar(
        summary, x="segment", y="default_rate_pct",
        color="segment",
        color_discrete_map=dict(zip(names, PALETTE_1)),
        text=summary["default_rate_pct"].round(1),
        error_y=summary["ci_high_pct"] - summary["default_rate_pct"],
        error_y_minus=summary["default_rate_pct"] - summary["ci_low_pct"]
    )
    fig.add_hline(
        y=overall_rate, line_dash="dash",
        line_color="#111827", annotation_text="Portfolio Average"
    )
    fig.update_layout(
        showlegend=False, xaxis_title="",
        yaxis_title="Default Rate (%)"
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Year-wise Default Rate Comparison")
    fig = px.line(
        by_year, markers=True,
        color_discrete_sequence=PALETTE_2,
        labels={"value": "Default Rate (%)",
                "issue_year": "Issue Year", "variable": "Segment"}
    )
    fig.update_traces(line_width=3, marker_size=8)
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Prime Credit records the lowest default rate across 2008–2018. "
        "Only years with at least 200 loans per segment are included. "
        "Loans from 2017–2018 have shorter repayment histories, which "
        "can make their observed default rates appear lower."
    )

# =========================================================
# 8. CLUSTER MAP
# =========================================================
with tab_map:
    st.write(
        "Each point represents a borrower. Nearby points indicate "
        "borrowers with similar characteristics."
    )

    c1, c2 = st.columns(2)
    method = c1.radio(
        "Dimensionality Reduction",
        ["PCA", "t-SNE", "UMAP"], horizontal=True
    )
    colour_by = c2.radio(
        "Display Colour By",
        ["Segment", "Repaid / Default"], horizontal=True
    )

    x_col, y_col = {
        "PCA": ("pca_1", "pca_2"),
        "t-SNE": ("tsne_1", "tsne_2"),
        "UMAP": ("umap_1", "umap_2")
    }[method]

    plot_data = sample.copy()
    plot_data["Outcome"] = plot_data["Default"].map(
        {0: "Repaid", 1: "Default"}
    )

    if colour_by == "Segment":
        fig = px.scatter(
            plot_data, x=x_col, y=y_col, color="cluster_name",
            color_discrete_map=colors,
            category_orders={"cluster_name": names},
            render_mode="webgl", opacity=0.65, height=600
        )
    else:
        fig = px.scatter(
            plot_data, x=x_col, y=y_col, color="Outcome",
            color_discrete_map={
                "Repaid": "#16a085", "Default": "#e74c3c"
            },
            render_mode="webgl", opacity=0.65, height=600
        )

    fig.update_traces(marker_size=5)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Some overlap between groups is normal because borrower "
        "characteristics change gradually rather than forming strict boundaries."
    )

# =========================================================
# 9. TRY A BORROWER
# =========================================================
with tab_try:
    st.warning(
        "Educational demonstration only. This tool identifies the closest "
        "borrower segment and displays its historical default rate. "
        "It does not approve or reject loans."
    )

    c1, c2, c3 = st.columns(3)
    income = c1.number_input(
        "Annual Income ($)", 5000, 1000000, 65000, 1000
    )
    dti = c2.number_input(
        "Debt-to-Income Ratio", 0.0, 100.0, 17.0, 0.5
    )
    loan = c3.number_input(
        "Loan Amount ($)", 500, 40000, 12000, 500
    )

    c1, c2, c3 = st.columns(3)
    fico = c1.slider("FICO Score", 600, 850, 690)
    emp_length = c2.selectbox(
        "Employment Length", list(model["emp_length_map"]), index=5
    )
    home = c3.selectbox(
        "Home Ownership", ["MORTGAGE", "RENT", "OWN"]
    )

    purpose = st.selectbox(
        "Loan Purpose",
        ["debt_consolidation", "credit_card", "home_improvement",
         "other", "major_purchase", "medical", "small_business",
         "car", "moving", "vacation", "house", "wedding",
         "renewable_energy", "educational"]
    )

    if st.button("🔍 Find My Borrower Segment", use_container_width=True):
        caps = model["caps"]

        person = pd.DataFrame([{
            "revenue": income,
            "dti_n": dti,
            "loan_amnt": loan,
            "fico_n": fico,
            "purpose": purpose,
            "home_ownership_n": home,
            "emp_years": model["emp_length_map"][emp_length]
        }])

        person["log_income"] = np.log1p(
            person["revenue"].clip(upper=caps["revenue"])
        )
        person["dti_n"] = person["dti_n"].clip(upper=caps["dti_n"])
        person["loan_to_income"] = (
            person["loan_amnt"] / person["revenue"]
        ).clip(upper=caps["loan_to_income"])

        person["purpose_group"] = np.where(
            person["purpose"].isin([
                "debt_consolidation", "credit_card", "home_improvement"
            ]),
            person["purpose"], "other"
        )

        scaled = model["preprocessor"].transform(
            person[model["feature_columns"]]
        )
        reduced = model["pca"].transform(scaled)
        predicted = model["kmeans"].predict(reduced)[0]
        segment_id = int(model["label_map"][predicted])
        segment = model["cluster_names"][segment_id]

        seg_row = summary[summary["segment"] == segment].iloc[0]

        st.success(f"Closest Borrower Segment: {segment}")
        st.metric(
            "Historical Default Rate",
            f"{seg_row['default_rate_pct']:.1f}%"
        )
        st.write(
            f"Historically, {seg_row['default_rate_pct']:.1f}% of "
            f"borrowers in this group defaulted, compared with the "
            f"overall portfolio rate of {overall_rate:.1f}%."
        )

# =========================================================
# 10. INSIGHTS
# =========================================================
with tab_insights:
    low = summary.loc[summary["default_rate_pct"].idxmin()]
    high = summary.loc[summary["default_rate_pct"].idxmax()]

    st.subheader("Key Findings")
    st.markdown(f"""
**1. Different repayment patterns:** Default rates range from
**{low['default_rate_pct']:.1f}%** in {low['segment']} to
**{high['default_rate_pct']:.1f}%** in {high['segment']}.
The higher rate is approximately
**{high['default_rate_pct'] / low['default_rate_pct']:.1f} times**
the lower rate.

**2. Affordability matters:** A good credit score alone does not
guarantee repayment. Stretched Low-Income borrowers have relatively
ordinary FICO scores but higher debt-to-income and loan-to-income ratios.

**3. Unusual does not always mean risky:** DBSCAN identifies unusual
borrowers, many of whom have exceptionally high incomes or credit scores.
These observations are outliers, but they are not automatically high-risk.
""")

    st.subheader("Possible Banking Applications")
    st.markdown("""
**1. Lending policies:** Banks could review loan-to-income limits more
carefully for the Stretched Low-Income segment.

**2. Customer-specific products:** Early-Career Renters could be offered
smaller entry-level loans, while Prime Credit customers may qualify for
competitive interest rates and faster services.

**3. Portfolio tracking:** Banks could monitor changes in the size of
each borrower segment every quarter to identify changing portfolio patterns.
""")

    st.subheader("Project Limitations")
    st.markdown("""
- **Cluster overlap:** A Silhouette score of approximately 0.15 indicates
  that the groups are not sharply separated. They represent general patterns,
  not strict categories.
- **Recent loan history:** Loans from 2017–2018 have had less time to
  develop defaults, so their observed rates may be lower.
- **Group-level analysis:** These segments describe borrower populations.
  A person's creditworthiness should never be judged using their segment alone.
""")

# =========================================================
# 11. FOOTER
# =========================================================
st.divider()
st.caption(
    "Credit Portfolio Segmentation | Unsupervised Machine Learning Project"
)
