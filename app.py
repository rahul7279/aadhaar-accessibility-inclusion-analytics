import streamlit as st
import pandas as pd
import glob
import matplotlib.pyplot as plt

# Try importing geopandas (only for local)
try:
    import geopandas as gpd
    GEO_AVAILABLE = True
except:
    GEO_AVAILABLE = False

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

    # Fallback (Cloud)
    if len(demo_files) == 0 or len(bio_files) == 0 or len(enr_files) == 0:
        st.warning("Using sample data (cloud demo mode)")

        demo_df = pd.DataFrame({
            "state": ["Bihar","Bihar","Bihar"],
            "district": ["Patna","Gaya","Muzaffarpur"]
        })

        bio_df = pd.DataFrame({
            "state": ["Bihar","Bihar","Bihar","Bihar"],
            "district": ["Patna","Patna","Gaya","Muzaffarpur"]
        })

        enr_df = pd.DataFrame({
            "state": ["Bihar","Bihar","Bihar","Bihar","Bihar"],
            "district": ["Patna","Gaya","Gaya","Muzaffarpur","Muzaffarpur"]
        })

        return demo_df, bio_df, enr_df

    # Real Data (Offline)
    demo_df = pd.concat([pd.read_csv(f) for f in demo_files], ignore_index=True)
    bio_df  = pd.concat([pd.read_csv(f) for f in bio_files], ignore_index=True)
    enr_df  = pd.concat([pd.read_csv(f) for f in enr_files], ignore_index=True)

    return demo_df, bio_df, enr_df


# LOAD
demo_df, bio_df, enr_df = load_data()

# -------------------------
# STATE SELECT
# -------------------------
states = sorted(demo_df["state"].dropna().unique())

selected_state = st.sidebar.selectbox(
    "Select State",
    states,
    index=states.index("Bihar") if "Bihar" in states else 0
)

# -------------------------
# FILTER
# -------------------------
demo_state = demo_df[demo_df["state"] == selected_state]
bio_state  = bio_df[bio_df["state"] == selected_state]
enr_state  = enr_df[enr_df["state"] == selected_state]

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
# ASI
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
# RISK
# -------------------------
inclusion_df["risk_score"] = (
    (inclusion_df["bio_usage"] - inclusion_df["enrolments"]) /
    (inclusion_df["demo_updates"] + 1)
)

def risk_label(score):
    if score > 1:
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
    inclusion_df[["district","asi_level","asi_score"]]
    .sort_values("asi_score", ascending=False),
    use_container_width=True
)

if st.checkbox("Show Full Risk Table"):
    st.dataframe(inclusion_df.sort_values("risk_score", ascending=False))

# -------------------------
# MAP (SMART HANDLING)
# -------------------------
if st.checkbox("Show Map"):

    if not GEO_AVAILABLE:
        st.warning("Map not available in cloud. Run locally for full geo view.")
    else:
        try:
            india_dist = gpd.read_file("shapefiles/gadm41_IND_2.shp")

            state_map = india_dist[
                india_dist["NAME_1"].str.lower() == selected_state.lower()
            ].copy()

            state_map["district"] = state_map["NAME_2"].str.lower().str.strip()
            inclusion_df["district"] = inclusion_df["district"].str.lower().str.strip()

            map_df = state_map.merge(inclusion_df, on="district", how="left")

            fig, ax = plt.subplots(figsize=(8,10))

            map_df.plot(
                column="asi_level",
                legend=True,
                cmap="RdYlGn_r",
                edgecolor="black",
                ax=ax
            )

            selected = map_df[map_df["district"] == district.lower()]
            if not selected.empty:
                selected.plot(ax=ax, color="orange")

            ax.axis("off")
            st.pyplot(fig)

        except Exception as e:
            st.error("Map loading failed. Check shapefile.")