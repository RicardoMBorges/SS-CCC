from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CCC Solvent System Planner",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# OPTIONAL LOGOS
# =========================================================
def load_logo(path_str: str):
    path = Path(path_str)
    if path.exists():
        return str(path)
    return None

LOGO_MAIN = load_logo("static/LAABio.png")
LOGO_SECONDARY = load_logo("static/IPPN.png")

# =========================================================
# HELPERS
# =========================================================
EXPECTED_SAMPLE_COLUMNS = ["sample_id"]
DEFAULT_SOLVENT_OPTIONS = [
    "Hexane",
    "Heptane",
    "Ethyl acetate",
    "Butanol",
    "Methanol",
    "Ethanol",
    "Isopropanol",
    "Acetonitrile",
    "Water",
    "Formic acid aqueous",
]


def smart_read_table(uploaded_file) -> pd.DataFrame:
    """Read CSV/TSV/TXT with basic delimiter detection."""
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8", errors="replace")

    candidates = [",", ";", "\t"]
    best_df = None
    best_cols = 0
    for sep in candidates:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            if df.shape[1] > best_cols:
                best_df = df.copy()
                best_cols = df.shape[1]
        except Exception:
            continue

    if best_df is None:
        raise ValueError("Could not parse the uploaded file.")

    best_df.columns = [str(c).strip() for c in best_df.columns]
    return best_df


@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def normalize_sample_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower_map = {c.lower().strip(): c for c in df.columns}

    if "sample_id" not in lower_map:
        # Use first column as sample_id when absent.
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "sample_id"})
    else:
        df = df.rename(columns={lower_map["sample_id"]: "sample_id"})

    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df = df[df["sample_id"] != ""].reset_index(drop=True)
    df["sample_index"] = np.arange(1, len(df) + 1)
    return df


def generate_five_systems(total_volume_ml: float) -> pd.DataFrame:
    """
    Placeholder design for 5 systems using four solvents.
    Rows sum to 1.0 and are later scaled to total volume.

    This is an initial exploration design and can be replaced later
    by a user-defined table or more advanced experimental design.
    """
    fractions = pd.DataFrame(
        {
            "System": ["S1", "S2", "S3", "S4", "S5"],
            "Solvent_1_frac": [0.25, 0.40, 0.15, 0.25, 0.25],
            "Solvent_2_frac": [0.25, 0.10, 0.35, 0.25, 0.25],
            "Solvent_3_frac": [0.25, 0.25, 0.25, 0.40, 0.10],
            "Solvent_4_frac": [0.25, 0.25, 0.25, 0.10, 0.40],
        }
    )

    for i in range(1, 5):
        fractions[f"Solvent_{i}_mL"] = fractions[f"Solvent_{i}_frac"] * total_volume_ml
        fractions[f"Solvent_{i}_uL"] = fractions[f"Solvent_{i}_mL"] * 1000

    fractions["Total_mL"] = total_volume_ml
    return fractions


def rename_solvent_columns(df: pd.DataFrame, solvent_names: List[str]) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    for i, solvent in enumerate(solvent_names, start=1):
        rename_map[f"Solvent_{i}_frac"] = f"{solvent}_frac"
        rename_map[f"Solvent_{i}_mL"] = f"{solvent}_mL"
        rename_map[f"Solvent_{i}_uL"] = f"{solvent}_uL"
    return df.rename(columns=rename_map)


def make_system_plot(df_systems: pd.DataFrame, solvent_names: List[str]) -> go.Figure:
    frac_cols = [f"{s}_frac" for s in solvent_names]
    plot_df = df_systems[["System"] + frac_cols].melt(
        id_vars="System", var_name="Solvent", value_name="Fraction"
    )
    plot_df["Percent"] = plot_df["Fraction"] * 100
    plot_df["Solvent"] = plot_df["Solvent"].str.replace("_frac", "", regex=False)

    fig = px.bar(
        plot_df,
        x="System",
        y="Percent",
        color="Solvent",
        barmode="stack",
        title="Composition of the 5 proposed solvent systems",
        hover_data={"Fraction": ":.3f", "Percent": ":.1f"},
    )
    fig.update_layout(
        xaxis_title="System",
        yaxis_title="Composition (%)",
        legend_title="Solvent",
    )
    return fig


def build_preparation_table(samples_df: pd.DataFrame, systems_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in samples_df.iterrows():
        sample_id = row["sample_id"]
        sample_index = int(row["sample_index"])
        batch_label = row.get("batch_label", "B01")
        for system in systems_df["System"]:
            tube_code = f"{batch_label}_SM{sample_index:03d}_{system}"
            vial_label = f"{sample_id}_{system}"
            records.append(
                {
                    "batch_label": batch_label,
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "system": system,
                    "tube_code": tube_code,
                    "label_text": vial_label,
                    "target_sample_mass_mg": 50,
                    "aliquot_volume_mL": 1.0,
                    "source_solution_concentration_mg_mL": 50.0,
                    "recommended_initial_mass_mg": 300,
                    "recommended_initial_solution_volume_mL": 6.0,
                }
            )
    return pd.DataFrame(records)


def build_metadata_table(
    samples_df: pd.DataFrame,
    systems_df: pd.DataFrame,
    solvent_names: List[str],
) -> pd.DataFrame:
    records = []
    for _, srow in samples_df.iterrows():
        batch_label = srow.get("batch_label", "B01")
        sample_id = srow["sample_id"]
        sample_index = int(srow["sample_index"])

        for _, sysrow in systems_df.iterrows():
            system = sysrow["System"]
            tube_code = f"{batch_label}_SM{sample_index:03d}_{system}"
            entry = {
                "batch_label": batch_label,
                "sample_id": sample_id,
                "sample_index": sample_index,
                "system": system,
                "tube_code": tube_code,
                "label_text": f"{sample_id}_{system}",
                "sample_mass_equivalent_mg": 50,
                "aliquot_volume_mL": 1.0,
            }
            for solvent in solvent_names:
                entry[f"{solvent}_mL"] = sysrow[f"{solvent}_mL"]
                entry[f"{solvent}_uL"] = sysrow[f"{solvent}_uL"]
            entry["total_solvent_volume_mL"] = sysrow["Total_mL"]
            records.append(entry)
    return pd.DataFrame(records)


def build_future_ot2_table(metadata_df: pd.DataFrame, solvent_names: List[str]) -> pd.DataFrame:
    out = metadata_df[["tube_code", "sample_id", "system", "label_text"]].copy()
    for solvent in solvent_names:
        out[f"{solvent}_uL"] = metadata_df[f"{solvent}_uL"]
    out["mix_after_addition"] = "TO_DEFINE"
    out["aspirate_height_strategy"] = "TO_DEFINE"
    out["dispense_height_strategy"] = "TO_DEFINE"
    return out


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## CCC Planner")
    st.caption("Experimental planning for solvent-system studies and future OT-2 export.")

    if LOGO_MAIN:
        st.image(LOGO_MAIN, use_container_width=True)
    if LOGO_SECONDARY:
        st.image(LOGO_SECONDARY, use_container_width=True)

    st.markdown("---")
    st.markdown("### Study setup")
    batch_size = st.number_input(
        "Batch size (number of samples in this run)",
        min_value=1,
        max_value=999,
        value=10,
        step=1,
    )
    batch_label = st.text_input("Batch label", value="B01")
    total_volume_ml = st.number_input(
        "Total solvent volume per system/tube (mL)",
        min_value=0.1,
        max_value=20.0,
        value=4.0,
        step=0.1,
    )

    st.markdown("### Solvent selection")
    solvent_1 = st.selectbox("Solvent 1", DEFAULT_SOLVENT_OPTIONS, index=0)
    solvent_2 = st.selectbox("Solvent 2", DEFAULT_SOLVENT_OPTIONS, index=2)
    solvent_3 = st.selectbox("Solvent 3", DEFAULT_SOLVENT_OPTIONS, index=4)
    solvent_4 = st.selectbox("Solvent 4", DEFAULT_SOLVENT_OPTIONS, index=8)
    solvent_names = [solvent_1, solvent_2, solvent_3, solvent_4]

    duplicated_solvents = len(set(solvent_names)) != 4
    if duplicated_solvents:
        st.warning("Please choose four different solvents.")

    st.markdown("---")
    st.markdown("### Future OT-2 section")
    st.caption("Reserved for future export settings.")
    st.text_input("OT-2 labware profile", value="TO_DEFINE")
    st.text_input("OT-2 pipette profile", value="TO_DEFINE")


# =========================================================
# MAIN HEADER
# =========================================================
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if LOGO_MAIN:
        st.image(LOGO_MAIN, use_container_width=True)
with col_title:
    st.title("CCC Solvent System Planner")
    st.markdown(
        "Design, document, and organize solvent-system studies for countercurrent chromatography "
        "with a future-ready structure for OT-2 automation."
    )

st.markdown("---")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1. Sample Upload",
        "2. Solvent Systems",
        "3. Sample Preparation",
        "4. Labels & Metadata",
        "5. Future OT-2 Export",
    ]
)

# =========================================================
# TAB 1 - SAMPLE UPLOAD
# =========================================================
with tab1:
    st.subheader("Upload sample list")
    with st.expander("What this tab does", expanded=True):
        st.markdown(
            "Upload a sample table to define which samples will enter the current batch. "
            "At minimum, the file should contain a column called `sample_id`. If it does not, "
            "the first column will be used as `sample_id`."
        )

    uploaded_file = st.file_uploader(
        "Upload CSV/TSV/TXT with your sample list",
        type=["csv", "tsv", "txt"],
    )

    if uploaded_file is not None:
        try:
            samples_df = smart_read_table(uploaded_file)
            samples_df = normalize_sample_table(samples_df)
            samples_df["batch_label"] = batch_label

            if len(samples_df) > batch_size:
                st.warning(
                    f"The uploaded file contains {len(samples_df)} samples, but the selected batch size is {batch_size}. "
                    f"Only the first {batch_size} samples will be used in this run."
                )
                samples_df = samples_df.head(batch_size).copy()
            elif len(samples_df) < batch_size:
                st.info(
                    f"The uploaded file contains {len(samples_df)} samples. The current run will use those {len(samples_df)} samples."
                )

            st.success("Sample table loaded successfully.")
            st.dataframe(samples_df, use_container_width=True)
            st.session_state["samples_df"] = samples_df
        except Exception as e:
            st.error(f"Could not read sample table: {e}")
    else:
        st.info("Upload a sample list to continue.")

# =========================================================
# COMMON OBJECTS
# =========================================================
samples_df = st.session_state.get("samples_df", pd.DataFrame())
can_build = (not duplicated_solvents) and (not samples_df.empty)

systems_df = pd.DataFrame()
renamed_systems_df = pd.DataFrame()
prep_df = pd.DataFrame()
metadata_df = pd.DataFrame()
future_ot2_df = pd.DataFrame()

if can_build:
    systems_df = generate_five_systems(total_volume_ml=total_volume_ml)
    renamed_systems_df = rename_solvent_columns(systems_df, solvent_names)
    prep_df = build_preparation_table(samples_df, renamed_systems_df)
    metadata_df = build_metadata_table(samples_df, renamed_systems_df, solvent_names)
    future_ot2_df = build_future_ot2_table(metadata_df, solvent_names)

# =========================================================
# TAB 2 - SOLVENT SYSTEMS
# =========================================================
with tab2:
    st.subheader("Proposed solvent systems")
    with st.expander("How these 5 systems are being generated", expanded=True):
        st.markdown(
            "This is an initial placeholder design for 5 solvent systems using 4 selected solvents. "
            "The current version distributes the compositions in a simple exploratory way so the app structure can be validated. "
            "Later, this section can be replaced by a custom input table, experimental design strategy, or HEMWat-style logic."
        )

    if not can_build:
        st.info("Upload samples and select four different solvents to generate the systems.")
    else:
        st.dataframe(renamed_systems_df, use_container_width=True)
        fig = make_system_plot(renamed_systems_df, solvent_names)
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download solvent systems table (CSV)",
            data=convert_df_to_csv(renamed_systems_df),
            file_name="ccc_solvent_systems.csv",
            mime="text/csv",
        )

# =========================================================
# TAB 3 - SAMPLE PREPARATION
# =========================================================
with tab3:
    st.subheader("Sample preparation instructions")
    with st.expander("Preparation logic", expanded=True):
        st.markdown(
            "The current workflow assumes: **300 mg** initial sample mass, preparation of a **50 mg/mL** stock solution, "
            "and transfer of **1.0 mL aliquots** to create five tubes, each containing **50 mg equivalent** of sample. "
            "These tubes are then intended for solvent addition and future OT-2 handling."
        )

    if not can_build:
        st.info("Preparation instructions will appear after loading samples and defining solvents.")
    else:
        st.markdown("### General preparation instructions")
        st.markdown(
            f"""
For each sample in batch **{batch_label}**:

1. Weigh **300 mg** of sample.
2. Dissolve to a final volume of **6.0 mL** to obtain **50 mg/mL**.
3. Prepare **five tubes** corresponding to systems **S1–S5**.
4. Transfer **1.0 mL** to each tube.
5. Each tube will contain **50 mg equivalent** of sample.
6. After this step, each tube should receive the solvent system defined in the solvent-systems tab.
"""
        )

        st.markdown("### Tube preparation plan")
        st.dataframe(prep_df, use_container_width=True)

        st.download_button(
            "Download preparation plan (CSV)",
            data=convert_df_to_csv(prep_df),
            file_name="ccc_sample_preparation_plan.csv",
            mime="text/csv",
        )

# =========================================================
# TAB 4 - LABELS & METADATA
# =========================================================
with tab4:
    st.subheader("Suggested labels and metadata")
    with st.expander("What is being created here", expanded=True):
        st.markdown(
            "This tab creates suggested tube codes, label text, and a metadata table joining the uploaded sample list "
            "with the generated solvent-system information."
        )

    if not can_build:
        st.info("Metadata will be generated after samples and solvent systems are available.")
    else:
        st.dataframe(metadata_df, use_container_width=True)

        st.download_button(
            "Download metadata table (CSV)",
            data=convert_df_to_csv(metadata_df),
            file_name="ccc_metadata.csv",
            mime="text/csv",
        )

# =========================================================
# TAB 5 - FUTURE OT-2 EXPORT
# =========================================================
with tab5:
    st.subheader("Future OT-2 import table")
    with st.expander("Status of this section", expanded=True):
        st.markdown(
            "This section is intentionally provisional. It already creates a future-facing table structure that can later be exported "
            "for OT-2 protocols after the exact labware, pipette, aspiration strategy, dispensing strategy, and mixing logic are finalized."
        )

    if not can_build:
        st.info("The future OT-2 table will be generated after the sample and solvent-system information is ready.")
    else:
        st.dataframe(future_ot2_df, use_container_width=True)

        st.download_button(
            "Download future OT-2 table (CSV)",
            data=convert_df_to_csv(future_ot2_df),
            file_name="ccc_future_ot2_import_table.csv",
            mime="text/csv",
        )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Initial app structure for CCC solvent-system planning, metadata generation, and future OT-2 integration."
)

