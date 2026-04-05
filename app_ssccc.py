from __future__ import annotations

import io
import os
import re
import zipfile
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

def expand_systems_to_phases(systems_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each solvent system (S1...S5) into two phases:
    superior (sup) and inferior (inf).

    Example:
        S1 -> S1_sup, S1_inf
    """
    records = []
    for _, row in systems_df.iterrows():
        base = row.to_dict()
        system_name = str(base["System"])

        for phase in ["sup", "inf"]:
            new_row = base.copy()
            new_row["base_system"] = system_name
            new_row["phase"] = phase
            new_row["System"] = f"{system_name}_{phase}"
            records.append(new_row)

    return pd.DataFrame(records)

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

        for _, sysrow in systems_df.iterrows():
            system = sysrow["System"]           # e.g. S1_sup
            base_system = sysrow["base_system"] # e.g. S1
            phase = sysrow["phase"]             # sup / inf

            tube_code = f"{batch_label}_SM{sample_index:03d}_{system}"
            vial_label = f"{sample_id}_{system}"

            records.append(
                {
                    "batch_label": batch_label,
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "base_system": base_system,
                    "phase": phase,
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
            system = sysrow["System"]            # S1_sup / S1_inf
            base_system = sysrow["base_system"]  # S1
            phase = sysrow["phase"]              # sup / inf

            tube_code = f"{batch_label}_SM{sample_index:03d}_{system}"

            entry = {
                "batch_label": batch_label,
                "sample_id": sample_id,
                "sample_index": sample_index,
                "base_system": base_system,
                "phase": phase,
                "system": system,
                "tube_code": tube_code,
                "label_text": f"{sample_id}_{system}",
                "sample_mass_equivalent_mg": 50,
                "aliquot_volume_mL": 1.0,
                "ATTRIBUTE_CCC": f"{base_system}_{'FS' if phase == 'sup' else 'FI'}",
                "HPLC_filename": "",
            }

            for solvent in solvent_names:
                entry[f"{solvent}_mL"] = sysrow[f"{solvent}_mL"]
                entry[f"{solvent}_uL"] = sysrow[f"{solvent}_uL"]

            entry["total_solvent_volume_mL"] = sysrow["Total_mL"]
            records.append(entry)

    return pd.DataFrame(records)


def build_future_ot2_table(metadata_df: pd.DataFrame, solvent_names: List[str]) -> pd.DataFrame:
    out = metadata_df[
        ["tube_code", "sample_id", "base_system", "phase", "system", "label_text", "ATTRIBUTE_CCC"]
    ].copy()

    for solvent in solvent_names:
        out[f"{solvent}_uL"] = metadata_df[f"{solvent}_uL"]

    out["mix_after_addition"] = "TO_DEFINE"
    out["aspirate_height_strategy"] = "TO_DEFINE"
    out["dispense_height_strategy"] = "TO_DEFINE"
    return out

# HPLC Import Helper
def parse_labsolutions_ascii(file_name: str, raw_bytes: bytes) -> pd.DataFrame:
    """
    Return DF with columns ['RT(min)', <sample_name>] from LabSolutions ASCII.
    Supports:
      - 2D ASCII: header 'R.Time (min)    Intensity'
      - PDA 3D ASCII: section [PDA 3D] with wavelength columns
    For 3D, this function extracts one wavelength trace automatically
    (default: nearest to 254 nm).
    """
    try:
        text = raw_bytes.decode("latin1", errors="ignore")
    except Exception:
        text = raw_bytes.decode("utf-8", errors="ignore")

    base = os.path.splitext(os.path.basename(file_name))[0]

    header_2d_re = re.compile(r"^R\.Time \(min\)\s+Intensity\s*$", flags=re.MULTILINE)
    header_3d_re = re.compile(r"^\[PDA 3D\]\s*$", flags=re.MULTILINE)

    m2d = header_2d_re.search(text)
    if m2d:
        table = text[m2d.start():]
        df = pd.read_csv(io.StringIO(table), sep=r"\s+", engine="python")
        df = df.iloc[:, :2].copy()
        df.columns = ["RT(min)", base]

        df["RT(min)"] = pd.to_numeric(
            df["RT(min)"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )
        df[base] = pd.to_numeric(
            df[base].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )

        df = df[df["RT(min)"].notna()].reset_index(drop=True)
        return df

    if header_3d_re.search(text):
        lines = text.splitlines()

        rt_header_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "R.Time (min)":
                rt_header_idx = i
                break

        if rt_header_idx is None:
            raise ValueError("Found PDA 3D block, but could not find 'R.Time (min)' header.")

        wl_line = lines[rt_header_idx + 1].strip()
        wl_tokens = re.split(r"\s+", wl_line)
        wl_tokens = [w for w in wl_tokens if w != ""]

        wavelengths = pd.to_numeric(pd.Series(wl_tokens), errors="coerce").dropna().tolist()
        if len(wavelengths) == 0:
            raise ValueError("Could not parse wavelength axis from PDA 3D file.")

        target_wl = 254
        wl_idx = int(np.argmin(np.abs(np.array(wavelengths) - target_wl)))

        data_rows = []
        for line in lines[rt_header_idx + 2:]:
            toks = re.split(r"\s+", line.strip())
            if len(toks) < 2:
                continue

            rt_val = pd.to_numeric(str(toks[0]).replace(",", "."), errors="coerce")
            if pd.isna(rt_val):
                continue

            numeric_vals = pd.to_numeric(
                pd.Series([str(x).replace(",", ".") for x in toks[1:]]),
                errors="coerce"
            ).tolist()

            if wl_idx < len(numeric_vals):
                inten = numeric_vals[wl_idx]
                data_rows.append([rt_val, inten])

        if not data_rows:
            raise ValueError("No valid RT/intensity pairs parsed from PDA 3D file.")

        df = pd.DataFrame(data_rows, columns=["RT(min)", base])
        df = df[df["RT(min)"].notna()].reset_index(drop=True)
        return df

    raise ValueError(
        "Unrecognized LabSolutions ASCII format. Expected either 2D or PDA 3D."
    )

def outer_join_rt(dfs: dict) -> pd.DataFrame:
    combined = None
    for _, df in dfs.items():
        combined = df if combined is None else combined.merge(df, on="RT(min)", how="outer")
    if combined is not None:
        combined = combined.sort_values("RT(min)").reset_index(drop=True)
    return combined

def resample_to_grid(df: pd.DataFrame, step: float, rt_min=None, rt_max=None):
    if df is None or df.empty or "RT(min)" not in df.columns:
        return None, None

    x = pd.to_numeric(df["RT(min)"], errors="coerce").values
    if x.size == 0 or np.all(np.isnan(x)):
        return None, None

    step = float(step) if step and float(step) > 0 else 0.02
    grid_min = float(rt_min) if rt_min is not None else float(np.nanmin(x))
    grid_max = float(rt_max) if rt_max is not None else float(np.nanmax(x))

    if grid_max <= grid_min:
        grid_max = grid_min + 0.01

    grid = np.arange(grid_min, grid_max + step / 2, step, dtype=float)
    out = {"RT(min)": grid}

    for c in df.columns:
        if c == "RT(min)":
            continue
        y = pd.to_numeric(df[c], errors="coerce").values
        y = pd.Series(y).interpolate(limit_direction="both").values
        out[c] = np.interp(grid, x, y, left=np.nan, right=np.nan)

    return pd.DataFrame(out), grid

def moving_average(arr: np.ndarray, win: int) -> np.ndarray:
    if win is None or win <= 1:
        return arr
    return pd.Series(arr).rolling(window=win, min_periods=1, center=True).mean().values


def baseline_subtract(arr: np.ndarray, method: str, param: float) -> np.ndarray:
    if method == "none":
        return arr
    if method == "median":
        return arr - np.nanmedian(arr)
    if method == "rolling_min":
        w = max(3, int(param))
        s = pd.Series(arr).rolling(window=w, min_periods=1, center=True).min().values
        return arr - s
    return arr


def normalize_trace(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return arr
    a = arr.copy()
    if mode == "max=1":
        m = np.nanmax(np.abs(a))
        return a / m if m else a
    if mode == "area=1":
        s = np.nansum(np.abs(a))
        return a / s if s else a
    if mode == "zscore":
        mu = np.nanmean(a)
        sd = np.nanstd(a)
        return (a - mu) / sd if sd else a
    return a


def preprocess_matrix(df: pd.DataFrame, smooth_win: int, baseline_method: str, baseline_param: float, norm_mode: str) -> pd.DataFrame:
    if df is None:
        return None
    out = df.copy()
    for c in out.columns:
        if c == "RT(min)":
            continue
        y = pd.to_numeric(out[c], errors="coerce").values
        y = moving_average(y, smooth_win)
        y = baseline_subtract(y, baseline_method, baseline_param)
        y = normalize_trace(y, norm_mode)
        out[c] = y
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "1. Sample Upload",
        "2. Solvent Systems",
        "3. Sample Preparation",
        "4. Labels & Metadata",
        "5. Future OT-2 Export",
        "6. HPLC Import",
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

    # Expanded table for actual produced tubes: S1_sup, S1_inf ... S5_sup, S5_inf
    phased_systems_df = expand_systems_to_phases(renamed_systems_df)

    prep_df = build_preparation_table(samples_df, phased_systems_df)
    metadata_df = build_metadata_table(samples_df, phased_systems_df, solvent_names)
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
"and transfer of **1.0 mL aliquots** to create **five initial system tubes (S1-S5)**, each containing "
"**50 mg equivalent** of sample. After solvent addition and phase separation, each system generates "
"**two phases**, resulting in **ten final analytical tubes per sample**: S1_inf, S1_sup, ..., S5_inf, S5_sup."
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
3. Prepare **five initial tubes** corresponding to systems **S1–S5**.
4. Transfer **1.0 mL** to each initial system tube.
5. Each initial tube will contain **50 mg equivalent** of sample.
6. After solvent addition and phase separation, each system will produce:
   - **one superior phase tube**
   - **one inferior phase tube**
7. Therefore, each sample will generate **10 final labeled phase tubes**:
   **S1_inf, S1_sup, S2_inf, S2_sup, ..., S5_inf, S5_sup**.
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

with tab6:
    st.subheader("HPLC Data Import")
    with st.expander("What this tab does", expanded=True):
        st.markdown(
            """
This tab imports LabSolutions ASCII HPLC files, builds a combined chromatogram matrix,
resamples chromatograms on a common RT grid, applies optional preprocessing, and displays
overlay, stacked, and heatmap visualizations.

This imported HPLC layer will later be connected to:
- **Keq calculations**
- **phase metadata (FI / FS)**
- **bioactivity correlation**
"""
        )

    st.markdown("### 1. Upload chromatograms (.txt)")
    hplc_uploads = st.file_uploader(
        "Upload LabSolutions ASCII chromatograms",
        type=["txt"],
        accept_multiple_files=True,
        key="hplc_uploads_tab6",
    )

    st.markdown("### 2. Processing options")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        grid_step_tab6 = st.number_input(
            "Uniform grid step (min)",
            value=0.02,
            min_value=0.001,
            step=0.001,
            format="%.3f",
            key="grid_step_tab6",
        )
    with c2:
        smooth_win_tab6 = st.number_input(
            "Smoothing window (pts)",
            value=1,
            min_value=1,
            step=1,
            key="smooth_win_tab6",
        )
    with c3:
        baseline_method_tab6 = st.selectbox(
            "Baseline",
            options=["none", "median", "rolling_min"],
            index=1,
            key="baseline_method_tab6",
        )
    with c4:
        baseline_param_tab6 = st.number_input(
            "Baseline param",
            value=101,
            min_value=3,
            step=2,
            key="baseline_param_tab6",
        )

    c5, c6 = st.columns(2)
    with c5:
        norm_mode_tab6 = st.selectbox(
            "Normalization",
            options=["none", "max=1", "area=1", "zscore"],
            index=2,
            key="norm_mode_tab6",
        )
    with c6:
        rt_range_tab6 = st.text_input(
            "RT range (min,max) or blank",
            value="",
            key="rt_range_tab6",
        )

    rt_min_tab6 = rt_max_tab6 = None
    if rt_range_tab6.strip():
        try:
            parts = [float(x) for x in rt_range_tab6.split(",")]
            if len(parts) == 2:
                rt_min_tab6, rt_max_tab6 = parts
        except Exception:
            st.warning("RT range not parsed. Use format like: 0.5,45")

    combined_hplc = None
    hplc_grid_df = None

    if hplc_uploads:
        parsed = {}
        report_rows = []

        for f in hplc_uploads:
            try:
                raw = f.getvalue()
                df = parse_labsolutions_ascii(f.name, raw)
                parsed[f.name] = df
                report_rows.append({"file": f.name, "status": "parsed", "rows": len(df)})
            except Exception as e:
                report_rows.append({"file": f.name, "status": f"error: {e}", "rows": 0})

        report_df = pd.DataFrame(report_rows)

        st.markdown("### 3. Parsing report")
        st.dataframe(report_df, use_container_width=True)

        combined_hplc = outer_join_rt(parsed) if parsed else None

        if combined_hplc is not None and not combined_hplc.empty:
            st.session_state["combined_hplc_tab6"] = combined_hplc

            st.markdown("### 4. Combined raw matrix")
            st.expander.dataframe(combined_hplc, use_container_width=True)

            hplc_grid_df, _ = resample_to_grid(
                combined_hplc,
                step=float(grid_step_tab6),
                rt_min=rt_min_tab6,
                rt_max=rt_max_tab6,
            )

            if hplc_grid_df is not None and not hplc_grid_df.empty:
                hplc_grid_df = preprocess_matrix(
                    hplc_grid_df,
                    int(smooth_win_tab6),
                    baseline_method_tab6,
                    float(baseline_param_tab6),
                    norm_mode_tab6,
                )
                st.session_state["processed_hplc_tab6"] = hplc_grid_df

                st.markdown("### 5. Processed HPLC matrix")
                st.dataframe(hplc_grid_df, use_container_width=True)

                st.markdown("### 6. Visualizations")
                plot_df = hplc_grid_df.melt(
                    id_vars="RT(min)",
                    var_name="Sample",
                    value_name="Intensity"
                ).dropna(subset=["Intensity"])

                t61, t62, t63 = st.tabs(["Overlay", "Stacked", "Heatmap"])

                with t61:
                    fig1 = px.line(
                        plot_df,
                        x="RT(min)",
                        y="Intensity",
                        color="Sample",
                        title="Overlay chromatograms"
                    )
                    fig1.update_layout(
                        xaxis_title="RT (min)",
                        yaxis_title="Intensity",
                        legend_title="Sample"
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                with t62:
                    samples_sorted = sorted([c for c in hplc_grid_df.columns if c != "RT(min)"])
                    stack_step = st.number_input(
                        "Stack offset",
                        value=2.0,
                        step=0.5,
                        key="stack_step_tab6",
                    )
                    offset_map = {s: i * stack_step for i, s in enumerate(samples_sorted)}
                    plot_df["Intensity_offset"] = plot_df.apply(
                        lambda r: r["Intensity"] + offset_map[r["Sample"]],
                        axis=1
                    )

                    fig2 = px.line(
                        plot_df,
                        x="RT(min)",
                        y="Intensity_offset",
                        color="Sample",
                        title="Stacked chromatograms"
                    )
                    fig2.update_layout(
                        xaxis_title="RT (min)",
                        yaxis_title=f"Intensity + offset (step={stack_step})"
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                with t63:
                    max_points = 5000
                    sub = hplc_grid_df
                    if len(hplc_grid_df) > max_points:
                        sub = hplc_grid_df.iloc[:: int(np.ceil(len(hplc_grid_df) / max_points)), :]

                    mat = sub[[c for c in sub.columns if c != "RT(min)"]].T.values

                    fig3 = go.Figure(
                        data=go.Heatmap(
                            z=mat,
                            x=sub["RT(min)"].values,
                            y=[c for c in sub.columns if c != "RT(min)"],
                            coloraxis="coloraxis"
                        )
                    )
                    fig3.update_layout(
                        title="Intensity heatmap",
                        xaxis_title="RT (min)",
                        yaxis_title="Sample",
                        coloraxis_colorscale="Viridis"
                    )
                    st.plotly_chart(fig3, use_container_width=True)

                st.markdown("### 7. Downloads")
                st.download_button(
                    "Download processed HPLC matrix (CSV)",
                    data=convert_df_to_csv(hplc_grid_df),
                    file_name="hplc_processed_matrix.csv",
                    mime="text/csv",
                )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Initial app structure for CCC solvent-system planning, metadata generation, and future OT-2 integration."
)

