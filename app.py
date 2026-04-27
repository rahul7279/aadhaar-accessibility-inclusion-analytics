import streamlit as st
import pandas as pd
import glob
import matplotlib.pyplot as plt

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Aadhaar Inclusion Analytics",
    layout="wide"
)

st.title("Aadhaar Accessibility & Inclusion Analytics")
st.caption("Interactive, Privacy-aware Decision Support System")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    demo_files = glob.glob("data/demographic/*.csv")
    bio_files  = glob.glob("data/biometric/*.csv")
    enr_files  = glob.glob("data/enrolment/*.csv")

    # 🔥 CLOUD SAFE FALLBACK
    if len(demo_files) == 0 or len(bio_files) == 0 or len(enr_files) == 0:
        st.warning("Using sample data (cloud demo mode)")

        demo_df = pd.DataFrame({
            "state": ["Bihar", "Bihar"],
            "district": ["Patna", "Gaya"]
        })

        bio_df = pd.DataFrame({
            "state": ["Bihar", "Bihar"],
            "district": ["Patna", "Gaya"]
        })

        enr_df = pd.DataFrame({
            "state": ["Bihar", "Bihar"],
            "district": ["Patna", "Gaya"]
        })

        return demo_df, bio_df, enr_df

    # REAL DATA
    demo_df = pd.concat([pd.read_csv(f) for f in demo_files], ignore_index=True)
    bio_df  = pd.concat([pd.read_csv(f) for f in bio_files], ignore_index=True)
    enr_df  = pd.concat([pd.read_csv(f) for f in enr_files], ignore_index=True)

    return demo_df, bio_df, enr_df


# 🔥 IMPORTANT CALL
demo_df, bio_df, enr_df = load_data()

# -------------------------
# STATE SELECTION
# -------------------------
states = sorted(demo_df["state"].dropna().unique())

selected_state = st.sidebar.selectbox(
    "Select State",
    states,
    index=states.index("Bihar") if "Bihar" in states else 0
)

# -------------------------
# FILTER DATA
# -------------------------
demo_state = demo_df[demo_df["state"] == selected_state].copy()
bio_state  = bio_df[bio_df["state"] == selected_state].copy()
enr_state  = enr_df[enr_df["state"] == selected_state].copy()

# -------------------------
# AGGREGATION
# -------------------------
demo_dist = demo_state.groupby("district").size().reset_index(name="demo_updates")
bio_dist  = bio_state.groupby("district").size().reset_index(name="bio_usage")
enr_dist  = enr_state.groupby("district").size().reset_index(name="enrolments")

inclusion_df = (
    demo_dist
    .merge(bio_dist, on="district", how="outer")
    .merge(enr_dist, on="district", how="outer")
    .fillna(0)
)

# -------------------------
# ASI (STRESS INDEX)
# -------------------------
inclusion_df["asi_score"] = (
    (inclusion_df["bio_usage"] / (inclusion_df["enrolments"] + 1)) +
    (inclusion_df["demo_updates"] / (inclusion_df["enrolments"] + 1))
)

def asi_level(score):
    if score > 3:
        return "Critical Stress"
    elif score > 1.5:
        return "High Stress"
    elif score > 0.7:
        return "Moderate Stress"
    else:
        return "Stable"

inclusion_df["asi_level"] = inclusion_df["asi_score"].apply(asi_level)

# -------------------------
# RISK SCORE
# -------------------------
inclusion_df["risk_score"] = (
    (inclusion_df["bio_usage"] - inclusion_df["enrolments"]) /
    (inclusion_df["demo_updates"] + 1)
)

def risk_label(score):
    if score > 1.0:
        return "High Risk"
    elif score > 0.3:
        return "Medium Risk"
    else:
        return "Low Risk"

inclusion_df["risk_category"] = inclusion_df["risk_score"].apply(risk_label)

# -------------------------
# DISTRICT SELECT
# -------------------------
district = st.sidebar.selectbox(
    "Select District",
    sorted(inclusion_df["district"].unique())
)

row = inclusion_df[inclusion_df["district"] == district].iloc[0]

# -------------------------
# METRICS
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Risk Category", row["risk_category"])
col2.metric("Risk Score", round(row["risk_score"], 2))
col3.metric("Enrolments", int(row["enrolments"]))

st.metric("ASI Level", row["asi_level"])

# -------------------------
# TABLE
# -------------------------
st.subheader("Aadhaar Service Early Warning System")

st.dataframe(
    inclusion_df[["district", "asi_level", "asi_score"]]
    .sort_values("asi_score", ascending=False),
    use_container_width=True
)

if st.checkbox("Show Full Risk Table"):
    st.dataframe(
        inclusion_df.sort_values("risk_score", ascending=False),
        use_container_width=True
    )

# -------------------------
# MAP (SAFE VERSION)
# -------------------------
if st.checkbox("Show Map (Demo)"):
    st.info("Geo-spatial map available in local version only.")