from __future__ import annotations

import io
import os
import re
import zipfile
import math
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple

DP_AVAILABLE = True
try:
    import data_processing_HPLC as dp
except Exception as e:
    DP_AVAILABLE = False
    DP_IMPORT_ERROR = e

ALIGN_AVAILABLE = True
try:
    from alignment_utils import alignment_controls, align_df
except Exception as e:
    ALIGN_AVAILABLE = False
    ALIGN_IMPORT_ERROR = e

try:
    from pyicoshift import icoshift  # noqa: F401
except Exception:
    icoshift = None

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
STATIC_DIR = Path(__file__).parent / "static"
LOGO_pyMETAflow_HPLC_PATH = STATIC_DIR / "PBGdiscovery_transp.png"
try:
    logo = Image.open(LOGO_pyMETAflow_HPLC_PATH)  # raises if missing
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("Logo not found at static/pyMETAflow_HPLC.png")

# ------------------ Sidebar: Uploads & Preferences ------------------
LOGO_PATH = STATIC_DIR / "LAABio.png"

try:
    logo = Image.open(LOGO_PATH)  # raises if missing
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("Logo not found at static/LAABio.png")


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
                "BioActivity_filename": "",
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
def parse_labsolutions_ascii(file_name: str, raw_bytes: bytes, target_wavelength: float = 254.0) -> pd.DataFrame:
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

        target_wl = float(target_wavelength)
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

def add_region_overlays(fig, regions: list[dict]):
    for i, reg in enumerate(regions, start=1):
        fig.add_vrect(
            x0=reg["rt_start"],
            x1=reg["rt_end"],
            fillcolor="skyblue",
            opacity=0.20,
            line_width=0,
            annotation_text=f"R{i}",
            annotation_position="top left",
        )
    return fig


def integrate_regions_from_df(hplc_df: pd.DataFrame, regions: list[dict]) -> pd.DataFrame:
    """
    Calculate AUC for each chromatogram column across user-defined RT regions.

    Returns a dataframe with one row per chromatogram/sample and one column per region.
    """
    rt = pd.to_numeric(hplc_df["RT(min)"], errors="coerce").values
    sample_cols = [c for c in hplc_df.columns if c != "RT(min)"]

    records = []
    for sample in sample_cols:
        y = pd.to_numeric(hplc_df[sample], errors="coerce").values
        row = {"HPLC_filename": sample}

        total_auc = 0.0
        for i, reg in enumerate(regions, start=1):
            mask = (rt >= reg["rt_start"]) & (rt <= reg["rt_end"])
            if np.sum(mask) >= 2:
                auc = float(np.trapz(y[mask], rt[mask]))
            else:
                auc = np.nan
            row[f"AUC_R{i}"] = auc
            total_auc += 0.0 if pd.isna(auc) else auc

        row["AUC_TOTAL_SELECTED"] = total_auc
        records.append(row)

    return pd.DataFrame(records)


def merge_auc_with_metadata(auc_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    meta = metadata_df.copy()
    meta["HPLC_filename"] = meta["HPLC_filename"].astype(str).str.replace(".txt", "", regex=False).str.strip()
    auc_df = auc_df.copy()
    auc_df["HPLC_filename"] = auc_df["HPLC_filename"].astype(str).str.replace(".txt", "", regex=False).str.strip()

    merged = pd.merge(meta, auc_df, on="HPLC_filename", how="left")
    return merged


def calculate_keq_from_metadata(merged_auc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pair FI and FS using ATTRIBUTE_CCC convention:
    e.g. S1_FI and S1_FS

    Calculates both FS/FI and FI/FS for each AUC region and total selected AUC.
    """
    auc_cols = [c for c in merged_auc_df.columns if c.startswith("AUC_R")] + ["AUC_TOTAL_SELECTED"]
    auc_cols = [c for c in auc_cols if c in merged_auc_df.columns]

    pair_records = []

    # base pair id = ATTRIBUTE_CCC without suffix _FI/_FS
    temp = merged_auc_df.copy()
    temp["pair_id"] = temp["ATTRIBUTE_CCC"].astype(str).str.replace("_FI", "", regex=False).str.replace("_FS", "", regex=False)

    for pair_id, g in temp.groupby("pair_id"):
        fi = g[g["ATTRIBUTE_CCC"].astype(str).str.endswith("_FI")]
        fs = g[g["ATTRIBUTE_CCC"].astype(str).str.endswith("_FS")]

        if fi.empty or fs.empty:
            continue

        fi_row = fi.iloc[0]
        fs_row = fs.iloc[0]

        out = {
            "pair_id": pair_id,
            "sample_id": fi_row.get("sample_id", ""),
            "base_system": fi_row.get("base_system", ""),
            "FI_tube_code": fi_row.get("tube_code", ""),
            "FS_tube_code": fs_row.get("tube_code", ""),
            "FI_file": fi_row.get("HPLC_filename", ""),
            "FS_file": fs_row.get("HPLC_filename", ""),
        }

        for col in auc_cols:
            fi_val = fi_row.get(col, np.nan)
            fs_val = fs_row.get(col, np.nan)

            out[f"{col}_FI"] = fi_val
            out[f"{col}_FS"] = fs_val
            out[f"{col}_FS_over_FI"] = fs_val / fi_val if pd.notna(fi_val) and fi_val != 0 else np.nan
            out[f"{col}_FI_over_FS"] = fi_val / fs_val if pd.notna(fs_val) and fs_val != 0 else np.nan

        pair_records.append(out)

    return pd.DataFrame(pair_records)

def get_hplc_filenames_from_processed_df(processed_hplc_df: pd.DataFrame) -> list[str]:
    if processed_hplc_df is None or processed_hplc_df.empty:
        return []
    return [c for c in processed_hplc_df.columns if c != "RT(min)"]

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## CCC Planner")
    st.caption("Experimental planning for solvent-system studies and future OT-2 export.")

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
st.title("CCC Solvent System Planner")
st.markdown(
    "Design, document, and organize solvent-system studies for countercurrent chromatography "
    "with a future-ready structure for OT-2 automation."
)

st.markdown("---")

# =========================================================
# TABS
# =========================================================
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "PROPOSED PIPELINE",
        "1. Sample Upload",
        "2. Solvent Systems",
        "3. Sample Preparation",
        "4. Labels & Metadata",
        "5. Future OT-2 Export",
        "6. HPLC / Metadata / BioActivity / STOCSY",
        "7. Data Integration / Keq",
    ]
)

# =========================================================
# TAB 0 - PROPOSED PIPELINE
# =========================================================
with tab0:
    st.subheader("PROPOSED PIPELINE")
    pipeline_PATH = STATIC_DIR / "pipeline.png"
    try:
        pipeline = Image.open(pipeline_PATH)  # raises if missing
        st.image(pipeline, use_container_width=True)
    except FileNotFoundError:
        st.warning("Pipeline not found at static/pipeline.png")


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
    st.subheader("HPLC / Metadata / BioActivity / STOCSY")
    st.info(
        "Upload here the UPDATED metadata file exported from tab 4 and edited by the user. "
        "This file must contain at least the columns 'HPLC_filename' and, for STOCSY with bioactivity, "
        "'BioActivity_filename'."
    )

    with st.expander("What this tab does", expanded=True):
        st.markdown(
            """
This tab reproduces the pyMETAflow_HPLC workflow inside the SS-CCC app.

Workflow:
1. Upload HPLC ASCII chromatograms
2. Upload metadata containing HPLC_filename and BioActivity_filename
3. Upload bioactivity table
4. Map chromatograms through metadata
5. Preprocess chromatograms
6. Align chromatograms using PAFFT / RAFFT / Icoshift
7. Run STOCSY using chromatographic data and bioactivity mapping
"""
        )

    st.markdown("### 1. HPLC input mode")

    hplc_mode_tab6 = st.radio(
        "Select HPLC data type",
        ["2D LabSolutions ASCII", "3D PDA ASCII"],
        index=0,
        key="hplc_mode_tab6",
    )

    target_wavelength_tab6 = 254.0
    if hplc_mode_tab6 == "3D PDA ASCII":
        target_wavelength_tab6 = st.number_input(
            "Target wavelength (nm)",
            min_value=190.0,
            max_value=800.0,
            value=254.0,
            step=1.0,
            key="target_wavelength_tab6",
        )
    # -------------------------------------------------
    # 1) Uploads
    # -------------------------------------------------
    st.markdown("### 1. Upload chromatograms (.txt)")
    hplc_uploads_tab6 = st.file_uploader(
        "Upload LabSolutions ASCII chromatograms",
        type=["txt"],
        accept_multiple_files=True,
        key="hplc_uploads_tab6_main",
    )

    st.markdown("### 2. Upload metadata")
    meta_file_tab6 = st.file_uploader(
        "Upload metadata file",
        type=["csv", "txt", "tsv"],
        key="meta_file_tab6",
    )

    st.markdown("### 3. Upload bioactivity")
    bio_file_tab6 = st.file_uploader(
        "Upload bioactivity file",
        type=["csv", "txt", "tsv"],
        key="bio_file_tab6",
    )

    # -------------------------------------------------
    # 2) Read metadata / bioactivity
    # -------------------------------------------------
    meta_df_tab6 = None
    if meta_file_tab6 is not None:
        try:
            meta_df_tab6 = smart_read_table(meta_file_tab6)
        except Exception as e:
            st.error(f"Failed to read metadata: {e}")

    if meta_df_tab6 is not None:
        with st.expander("Metadata (head)", expanded=False):
            st.dataframe(meta_df_tab6.head(10), use_container_width=True)
            st.session_state["uploaded_metadata_tab6"] = meta_df_tab6.copy()

    bio_df_tab6 = None
    if bio_file_tab6 is not None:
        try:
            bio_df_tab6 = smart_read_table(bio_file_tab6)
        except Exception as e:
            st.error(f"Failed to read bioactivity: {e}")

    if bio_df_tab6 is not None:
        with st.expander("Bioactivity (head)", expanded=False):
            st.dataframe(bio_df_tab6.head(10), use_container_width=True)

    # -------------------------------------------------
    # 3) Metadata column mapping
    # -------------------------------------------------
    # Fixed metadata column names expected in the UPDATED metadata file
    col_sample = "sample_id"
    col_hplc = "HPLC_filename"
    col_bio = "BioActivity_filename"

    if meta_df_tab6 is not None:
        required_meta_cols = [col_sample, col_hplc]
        missing_meta_cols = [c for c in required_meta_cols if c not in meta_df_tab6.columns]

        if missing_meta_cols:
            st.error(
                f"Uploaded metadata is missing required columns: {missing_meta_cols}"
            )
            st.stop()

        if col_bio not in meta_df_tab6.columns:
            st.warning(
                "Column 'BioActivity_filename' was not found in the uploaded metadata. "
                "Bioactivity-linked STOCSY will not work until this column is added."
            )

        with st.expander("Metadata column requirements", expanded=False):
            st.markdown(
                """
The uploaded metadata must be the UPDATED file exported from tab 4 and edited by the user.

Required columns:
- `sample_id`
- `HPLC_filename`

Optional but needed for STOCSY with bioactivity:
- `BioActivity_filename`

Important:
- `HPLC_filename` must match the imported ASCII file stem
- do not use `.txt` in the mapping unless you use it consistently
"""
            )

    # -------------------------------------------------
    # 4) Parse HPLC uploads
    # -------------------------------------------------
    combined_tab6 = None

    if hplc_uploads_tab6:
        parsed = {}
        report_rows = []

        for f in hplc_uploads_tab6:
            try:
                raw = f.getvalue()
                if hplc_mode_tab6 == "3D PDA ASCII":
                    df = parse_labsolutions_ascii(
                        f.name,
                        raw,
                        target_wavelength=float(target_wavelength_tab6),
                    )
                else:
                    df = parse_labsolutions_ascii(f.name, raw)
                parsed[f.name] = df
                report_rows.append({"file": f.name, "status": "parsed", "rows": len(df)})
            except Exception as e:
                report_rows.append({"file": f.name, "status": f"error: {e}", "rows": 0})

        report_df = pd.DataFrame(report_rows)

        with st.expander("Parsing report", expanded=False):
            st.dataframe(report_df, use_container_width=True)

        combined_tab6 = outer_join_rt(parsed) if parsed else None

        if combined_tab6 is not None and not combined_tab6.empty:
            with st.expander("Combined raw matrix", expanded=False):
                st.dataframe(combined_tab6, use_container_width=True)

            imported_hplc_names = [c for c in combined_tab6.columns if c != "RT(min)"]

            with st.expander("Imported HPLC filenames detected in the combined matrix", expanded=False):
                st.markdown(
                    """
These are the chromatogram names detected from the uploaded ASCII files.

Write these exact names in the column `HPLC_filename`
of the UPDATED metadata file.
"""
                )
                st.dataframe(
                    pd.DataFrame({"HPLC_filename_detected": imported_hplc_names}),
                    use_container_width=True,
                )

            # ---------------------------------------------
            # Raw chromatogram preview (independent of STOCSY)
            # ---------------------------------------------
            st.markdown("### Raw chromatograms")

            plot_raw = combined_tab6.melt(
                id_vars="RT(min)",
                var_name="Sample",
                value_name="Intensity"
            ).dropna(subset=["Intensity"])

            raw_tab1, raw_tab2, raw_tab3 = st.tabs(["Overlay", "Stacked", "Heatmap"])

            with raw_tab1:
                fig_raw_overlay = px.line(
                    plot_raw,
                    x="RT(min)",
                    y="Intensity",
                    color="Sample",
                    title="Raw imported chromatograms"
                )
                st.plotly_chart(fig_raw_overlay, use_container_width=True)

            with raw_tab2:
                samples_sorted_raw = sorted([c for c in combined_tab6.columns if c != "RT(min)"])
                raw_stack_step = st.number_input(
                    "Raw stack offset",
                    value=2.0,
                    step=0.5,
                    key="raw_stack_step_tab6",
                )
                offset_map_raw = {s: i * raw_stack_step for i, s in enumerate(samples_sorted_raw)}
                plot_raw["Intensity_offset"] = plot_raw.apply(
                    lambda r: r["Intensity"] + offset_map_raw[r["Sample"]],
                    axis=1
                )

                fig_raw_stack = px.line(
                    plot_raw,
                    x="RT(min)",
                    y="Intensity_offset",
                    color="Sample",
                    title="Raw stacked chromatograms"
                )
                st.plotly_chart(fig_raw_stack, use_container_width=True)

            with raw_tab3:
                max_points_raw = 5000
                sub_raw = combined_tab6
                if len(combined_tab6) > max_points_raw:
                    sub_raw = combined_tab6.iloc[:: int(np.ceil(len(combined_tab6) / max_points_raw)), :]

                mat_raw = sub_raw[[c for c in sub_raw.columns if c != "RT(min)"]].T.values
                fig_raw_heat = go.Figure(
                    data=go.Heatmap(
                        z=mat_raw,
                        x=sub_raw["RT(min)"].values,
                        y=[c for c in sub_raw.columns if c != "RT(min)"],
                        coloraxis="coloraxis"
                    )
                )
                fig_raw_heat.update_layout(
                    title="Raw intensity heatmap",
                    xaxis_title="RT (min)",
                    yaxis_title="Sample",
                    coloraxis_colorscale="Viridis"
                )
                st.plotly_chart(fig_raw_heat, use_container_width=True)

    # -------------------------------------------------
    # 5) Preprocessing
    # -------------------------------------------------
    df_grid_tab6 = None

    if combined_tab6 is not None and "RT(min)" in combined_tab6.columns:
        st.markdown("### 4. Preprocessing")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            grid_step_tab6 = st.number_input(
                "Uniform grid step (min)",
                value=0.02,
                min_value=0.001,
                step=0.001,
                format="%.3f",
                key="grid_step_tab6_main",
            )
        with c2:
            smooth_win_tab6 = st.number_input(
                "Smoothing window (pts)",
                value=1,
                min_value=1,
                step=1,
                key="smooth_win_tab6_main",
            )
        with c3:
            baseline_method_tab6 = st.selectbox(
                "Baseline",
                ["none", "median", "rolling_min"],
                index=1,
                key="baseline_method_tab6_main",
            )
        with c4:
            baseline_param_tab6 = st.number_input(
                "Baseline param",
                value=101,
                min_value=3,
                step=2,
                key="baseline_param_tab6_main",
            )

        c5, c6 = st.columns(2)
        with c5:
            norm_mode_tab6 = st.selectbox(
                "Normalization",
                ["none", "max=1", "area=1", "zscore"],
                index=2,
                key="norm_mode_tab6_main",
            )
        with c6:
            rt_range_tab6 = st.text_input(
                "RT range (min,max) or blank",
                value="",
                key="rt_range_tab6_main",
            )

        rt_min = rt_max = None
        if rt_range_tab6.strip():
            try:
                parts = [float(x) for x in rt_range_tab6.split(",")]
                if len(parts) == 2:
                    rt_min, rt_max = parts
            except Exception:
                st.warning("RT range not parsed. Use format like: 0.5,45")

        df_grid_tab6, _ = resample_to_grid(
            combined_tab6,
            step=float(grid_step_tab6),
            rt_min=rt_min,
            rt_max=rt_max,
        )

        if df_grid_tab6 is not None and not df_grid_tab6.empty:
            df_grid_tab6 = preprocess_matrix(
                df_grid_tab6,
                int(smooth_win_tab6),
                baseline_method_tab6,
                float(baseline_param_tab6),
                norm_mode_tab6,
            )

            st.session_state["tab6_preprocessed_df"] = df_grid_tab6

            with st.expander("Processed HPLC matrix", expanded=False):
                st.dataframe(df_grid_tab6, use_container_width=True)

            plot_processed = df_grid_tab6.melt(
                id_vars="RT(min)",
                var_name="Sample",
                value_name="Intensity"
            ).dropna(subset=["Intensity"])

            fig_processed = px.line(
                plot_processed,
                x="RT(min)",
                y="Intensity",
                color="Sample",
                title="Processed chromatograms before alignment"
            )
            st.plotly_chart(fig_processed, use_container_width=True)

    # -------------------------------------------------
    # 6) Alignment
    # -------------------------------------------------
    df_aligned_tab6 = None
    align_source_tab6 = st.session_state.get("tab6_preprocessed_df", None)

    if align_source_tab6 is not None and not align_source_tab6.empty:
        st.markdown("### 5. Alignment")

        st.markdown("**Icoshift:** Interval correlation optimized shifting.")
        st.markdown("**PAFFT / RAFFT:** FFT-based alignment methods for RT correction.")

        # IMPORTANT: do NOT include RT(min) as a sample for alignment
        sample_names = [c for c in align_source_tab6.columns if c != "RT(min)"]

        with st.expander("Pre-alignment HPLC matrix", expanded=False):
            st.dataframe(align_source_tab6, use_container_width=True)

        # Quick pre-alignment overlay for debugging
        plot_pre = align_source_tab6.melt(
            id_vars="RT(min)",
            var_name="Sample",
            value_name="Intensity"
        ).dropna(subset=["Intensity"])

        fig_pre = px.line(
            plot_pre,
            x="RT(min)",
            y="Intensity",
            color="Sample",
            title="Pre-alignment chromatograms"
        )
        st.plotly_chart(fig_pre, use_container_width=True)

        method, params = alignment_controls(align_source_tab6, sample_names=sample_names)
        df_aligned_tab6 = align_df(align_source_tab6, method, **params)

        if df_aligned_tab6 is not None and not df_aligned_tab6.empty:
            st.session_state["tab6_aligned_df"] = df_aligned_tab6

            with st.expander("Post-alignment HPLC matrix", expanded=False):
                st.dataframe(df_aligned_tab6, use_container_width=True)

            # Diagnostics: if variance is ~0, alignment flattened the signal
            sample_cols_aligned = [c for c in df_aligned_tab6.columns if c != "RT(min)"]
            var_summary = pd.DataFrame({
                "Sample": sample_cols_aligned,
                "Variance": [pd.to_numeric(df_aligned_tab6[c], errors="coerce").var() for c in sample_cols_aligned],
                "Min": [pd.to_numeric(df_aligned_tab6[c], errors="coerce").min() for c in sample_cols_aligned],
                "Max": [pd.to_numeric(df_aligned_tab6[c], errors="coerce").max() for c in sample_cols_aligned],
            })

            with st.expander("Aligned signal diagnostics", expanded=False):
                st.dataframe(var_summary, use_container_width=True)

            plot_df = df_aligned_tab6.melt(
                id_vars="RT(min)",
                var_name="Sample",
                value_name="Intensity"
            ).dropna(subset=["Intensity"])

            t61, t62, t63 = st.tabs(["Overlay", "Stacked", "Heatmap"])

            with t61:
                fig_overlay = px.line(
                    plot_df,
                    x="RT(min)",
                    y="Intensity",
                    color="Sample",
                    title="Aligned chromatograms"
                )
                st.plotly_chart(fig_overlay, use_container_width=True)

            with t62:
                samples_sorted = sorted([c for c in df_aligned_tab6.columns if c != "RT(min)"])
                stack_step = st.number_input(
                    "Stack offset",
                    value=2.0,
                    step=0.5,
                    key="stack_step_tab6_aligned",
                )
                offset_map = {s: i * stack_step for i, s in enumerate(samples_sorted)}
                plot_df["Intensity_offset"] = plot_df.apply(
                    lambda r: r["Intensity"] + offset_map[r["Sample"]],
                    axis=1
                )

                fig_stack = px.line(
                    plot_df,
                    x="RT(min)",
                    y="Intensity_offset",
                    color="Sample",
                    title="Stacked aligned chromatograms"
                )
                st.plotly_chart(fig_stack, use_container_width=True)

            with t63:
                max_points = 5000
                sub = df_aligned_tab6
                if len(df_aligned_tab6) > max_points:
                    sub = df_aligned_tab6.iloc[:: int(np.ceil(len(df_aligned_tab6) / max_points)), :]

                mat = sub[[c for c in sub.columns if c != "RT(min)"]].T.values
                fig_heat = go.Figure(
                    data=go.Heatmap(
                        z=mat,
                        x=sub["RT(min)"].values,
                        y=[c for c in sub.columns if c != "RT(min)"],
                        coloraxis="coloraxis"
                    )
                )
                fig_heat.update_layout(
                    title="Aligned intensity heatmap",
                    xaxis_title="RT (min)",
                    yaxis_title="Sample",
                    coloraxis_colorscale="Viridis"
                )
                st.plotly_chart(fig_heat, use_container_width=True)


            with t61:
                fig_overlay = px.line(
                    plot_df,
                    x="RT(min)",
                    y="Intensity",
                    color="Sample",
                    title="Aligned chromatograms"
                )
                st.plotly_chart(fig_overlay, use_container_width=True)

            with t62:
                samples_sorted = sorted([c for c in df_aligned_tab6.columns if c != "RT(min)"])
                stack_step = st.number_input(
                    "Stack offset",
                    value=2.0,
                    step=0.5,
                    key="stack_step_tab6_aligned",
                )
                offset_map = {s: i * stack_step for i, s in enumerate(samples_sorted)}
                plot_df["Intensity_offset"] = plot_df.apply(
                    lambda r: r["Intensity"] + offset_map[r["Sample"]],
                    axis=1
                )

                fig_stack = px.line(
                    plot_df,
                    x="RT(min)",
                    y="Intensity_offset",
                    color="Sample",
                    title="Stacked aligned chromatograms"
                )
                st.plotly_chart(fig_stack, use_container_width=True)

            with t63:
                max_points = 5000
                sub = df_aligned_tab6
                if len(df_aligned_tab6) > max_points:
                    sub = df_aligned_tab6.iloc[:: int(np.ceil(len(df_aligned_tab6) / max_points)), :]

                mat = sub[[c for c in sub.columns if c != "RT(min)"]].T.values
                fig_heat = go.Figure(
                    data=go.Heatmap(
                        z=mat,
                        x=sub["RT(min)"].values,
                        y=[c for c in sub.columns if c != "RT(min)"],
                        coloraxis="coloraxis"
                    )
                )
                fig_heat.update_layout(
                    title="Aligned intensity heatmap",
                    xaxis_title="RT (min)",
                    yaxis_title="Sample",
                    coloraxis_colorscale="Viridis"
                )
                st.plotly_chart(fig_heat, use_container_width=True)

    if combined_tab6 is not None and meta_df_tab6 is not None:
        imported_hplc_names = set([c for c in combined_tab6.columns if c != "RT(min)"])
        metadata_hplc_names = set(
            meta_df_tab6["HPLC_filename"]
            .astype(str)
            .str.replace(".txt", "", regex=False)
            .str.strip()
        )

        matched_hplc = sorted(imported_hplc_names & metadata_hplc_names)
        missing_in_metadata = sorted(imported_hplc_names - metadata_hplc_names)
        missing_in_imports = sorted(metadata_hplc_names - imported_hplc_names)

        with st.expander("HPLC filename matching diagnostics", expanded=False):
            st.markdown(f"Matched HPLC filenames: **{len(matched_hplc)}**")
            st.markdown(f"Imported HPLC filenames missing in metadata: **{len(missing_in_metadata)}**")
            st.markdown(f"Metadata HPLC filenames missing in uploads: **{len(missing_in_imports)}**")

            if matched_hplc:
                st.dataframe(pd.DataFrame({"matched": matched_hplc}), use_container_width=True)

            if missing_in_metadata:
                st.dataframe(pd.DataFrame({"missing_in_metadata": missing_in_metadata}), use_container_width=True)

            if missing_in_imports:
                st.dataframe(pd.DataFrame({"missing_in_imports": missing_in_imports}), use_container_width=True)
    
    # -------------------------------------------------
    # 7) STOCSY
    # -------------------------------------------------
    df_aligned_tab6 = st.session_state.get("tab6_aligned_df", None)

    if df_aligned_tab6 is not None and meta_df_tab6 is not None and bio_df_tab6 is not None:
        st.markdown("### 6. STOCSY")

        use_bio_driver = st.checkbox(
            "Use BioActivity fusion and drive STOCSY with BioAct",
            value=True,
            key="use_bio_driver_tab6",
        )

        if use_bio_driver and all(c in meta_df_tab6.columns for c in [col_sample, col_hplc, col_bio]):
            try:
                # -----------------------------
                # Prepare aligned HPLC matrix
                # -----------------------------
                LC = df_aligned_tab6.drop(columns="RT(min)").copy()
                RT = pd.to_numeric(df_aligned_tab6["RT(min)"], errors="coerce").reset_index(drop=True)

                hplc_columns = (
                    pd.Index(LC.columns)
                    .astype(str)
                    .str.replace(".txt", "", regex=False)
                    .str.strip()
                    .tolist()
                )
                LC.columns = hplc_columns

                # -----------------------------
                # Prepare metadata mapping
                # -----------------------------
                meta_tmp = meta_df_tab6.copy()
                meta_tmp[col_sample] = meta_tmp[col_sample].astype(str).str.strip()
                meta_tmp[col_hplc] = (
                    meta_tmp[col_hplc]
                    .astype(str)
                    .str.replace(".txt", "", regex=False)
                    .str.strip()
                )
                meta_tmp[col_bio] = (
                    meta_tmp[col_bio]
                    .astype(str)
                    .str.replace(".txt", "", regex=False)
                    .str.strip()
                )

                # Keep only rows with all required mappings filled
                meta_tmp = meta_tmp[
                    (meta_tmp[col_sample] != "")
                    & (meta_tmp[col_hplc] != "")
                    & (meta_tmp[col_bio] != "")
                ].copy()

                # -----------------------------
                # Prepare bioactivity table
                # -----------------------------
                bio_df_tmp = bio_df_tab6.copy()
                bio_df_tmp.columns = (
                    pd.Index(bio_df_tmp.columns)
                    .astype(str)
                    .str.replace(".txt", "", regex=False)
                    .str.strip()
                )

                bio_cols = bio_df_tmp.columns.tolist()

                # Optional first non-data column removal if needed
                metadata_bio_names = meta_tmp[col_bio].tolist()
                if not all(b in bio_cols for b in metadata_bio_names):
                    if bio_df_tmp.shape[1] > 1:
                        bio_df_tmp = bio_df_tmp.iloc[:, 1:].copy()
                        bio_df_tmp.columns = (
                            pd.Index(bio_df_tmp.columns)
                            .astype(str)
                            .str.replace(".txt", "", regex=False)
                            .str.strip()
                        )
                        bio_cols = bio_df_tmp.columns.tolist()

                # -----------------------------
                # Match metadata to HPLC + BioActivity
                # -----------------------------
                samples_ok = []
                hplc_cols_needed = []
                bio_cols_needed = []

                for _, row in meta_tmp.iterrows():
                    sample_name = row[col_sample]
                    hplc_name = row[col_hplc]
                    bio_name = row[col_bio]

                    if (hplc_name in hplc_columns) and (bio_name in bio_cols):
                        samples_ok.append(sample_name)
                        hplc_cols_needed.append(hplc_name)
                        bio_cols_needed.append(bio_name)

                if len(samples_ok) == 0:
                    st.error(
                        "No overlapping entries were found between aligned HPLC columns, "
                        "metadata HPLC_filename, and bioactivity columns."
                    )
                else:
                    # -----------------------------
                    # Reorder HPLC matrix by metadata mapping
                    # -----------------------------
                    LC_ord = LC[hplc_cols_needed].copy()
                    LC_ord.columns = samples_ok

                    # -----------------------------
                    # Pick one numeric bioactivity row
                    # -----------------------------
                    picked = None
                    picked_row_idx = None

                    for r in range(bio_df_tmp.shape[0]):
                        vals = pd.to_numeric(
                            bio_df_tmp[bio_cols_needed].iloc[r],
                            errors="coerce"
                        )
                        if vals.notna().mean() > 0.8:
                            picked = vals.values.astype(float)
                            picked_row_idx = r
                            break

                    if picked is None:
                        vals = pd.to_numeric(
                            bio_df_tmp[bio_cols_needed].iloc[0],
                            errors="coerce"
                        )
                        picked = vals.values.astype(float)
                        picked_row_idx = 0

                    BioActdata = pd.DataFrame([picked], columns=bio_cols_needed)
                    BioActdata.rename(
                        columns={old: new for old, new in zip(bio_cols_needed, samples_ok)},
                        inplace=True
                    )
                    BioActdata = BioActdata[samples_ok]

                    # -----------------------------
                    # Merge HPLC + BioActivity
                    # -----------------------------
                    MergeDF = pd.concat([LC_ord, BioActdata], ignore_index=True)

                    gap = float(RT.values[-1] - RT.values[-2]) if len(RT) >= 2 else 0.01
                    new_point = float(RT.values[-1]) + (gap if gap > 0 else 0.01)
                    new_axis = pd.concat([RT, pd.Series([new_point])], ignore_index=True)

                    st.success(
                        f"Merged HPLC ({LC_ord.shape[0]} RT points) + BioAct (1 row) "
                        f"for {len(samples_ok)} mapped sample(s)."
                    )

                    with st.expander("STOCSY mapping diagnostics", expanded=False):
                        diag_df = pd.DataFrame({
                            "sample_id": samples_ok,
                            "HPLC_filename_used": hplc_cols_needed,
                            "BioActivity_filename_used": bio_cols_needed,
                        })
                        st.dataframe(diag_df, use_container_width=True)
                        st.caption(f"Bioactivity row used from uploaded table: row index {picked_row_idx}")

                    # -----------------------------
                    # STOCSY settings
                    # -----------------------------
                    target_rt = st.number_input(
                        "Target RT (min)",
                        value=11.25,
                        step=0.05,
                        format="%.2f",
                        key="target_rt_tab6",
                    )

                    stocsy_model = st.selectbox(
                        "Model",
                        [
                            "linear",
                            "exponential",
                            "sinusoidal",
                            "sigmoid",
                            "gaussian",
                            "fft",
                            "polynomial",
                            "piecewise",
                            "skewed_gauss",
                        ],
                        index=0,
                        key="stocsy_model_tab6",
                    )

                    if st.button("Run STOCSY", key="run_stocsy_tab6"):
                        # Important:
                        # the appended last point corresponds to the BioActivity driver
                        target_for_run = float(new_axis.values[-1])

                        corr = covar = None

                        if hasattr(dp, "STOCSY_LC_mode"):
                            try:
                                corr, covar = dp.STOCSY_LC_mode(
                                    target_for_run,
                                    MergeDF,
                                    new_axis,
                                    mode=stocsy_model
                                )
                            except Exception as e:
                                st.warning(f"dp.STOCSY_LC_mode failed ({e}). Falling back to stocsy_linear().")

                        if corr is None or covar is None:
                            corr, covar = stocsy_linear(target_for_run, MergeDF, new_axis)

                        res = pd.DataFrame(
                            {
                                "RT(min)": new_axis.values,
                                "Correlation": corr,
                                "Covariance": covar,
                            }
                        )

                        with st.expander("STOCSY table", expanded=False):
                            st.dataframe(res, use_container_width=True)

                        figc = px.scatter(
                            res,
                            x="RT(min)",
                            y="Covariance",
                            color="Correlation",
                            color_continuous_scale="Jet",
                            render_mode="webgl",
                            title="STOCSY: covariance colored by correlation",
                        )
                        figc.add_trace(
                            go.Scatter(
                                x=res["RT(min)"],
                                y=res["Covariance"],
                                mode="lines",
                                line=dict(width=1),
                                name="Covariance",
                            )
                        )
                        st.plotly_chart(figc, use_container_width=True)

                        st.download_button(
                            "Download STOCSY table (CSV)",
                            data=convert_df_to_csv(res),
                            file_name="stocsy_tab6.csv",
                            mime="text/csv",
                        )

            except Exception as e:
                st.error(f"STOCSY setup failed: {e}")

        else:
            st.info(
                "STOCSY with bioactivity requires sample_id, HPLC_filename, and BioActivity_filename in the uploaded metadata."
            )

with tab7:
    st.subheader("Data Integration / Keq")
    with st.expander("What this tab does", expanded=True):
        st.markdown(
            """
This tab integrates processed HPLC chromatograms with the metadata table.

Workflow:
1. Define RT regions for integration
2. Calculate AUC for each chromatogram
3. Merge AUC values with metadata
4. Pair FI and FS chromatograms
5. Calculate both:
   - FS / FI
   - FI / FS
"""
        )
        st.markdown("### Metadata-to-HPLC mapping")

        with st.expander("How HPLC filename mapping works", expanded=False):
            st.markdown(
                """
The **ss_ccc metadata table** is treated as the master metadata because it already contains
the CCC structure generated by the app itself, including:

- `sample_id`
- `base_system`
- `phase`
- `system`
- `tube_code`
- `ATTRIBUTE_CCC`

To connect chromatographic data to this table, each row must optionally receive an
`HPLC_filename` value.

### What should go in `HPLC_filename`
This field should contain the **chromatogram name exactly as imported in Tab 6**,
usually the file stem without `.txt`.

Examples:
- `SMPL001_S1_sup`
- `SMPL001_S1_inf`

### Why this is needed
The app calculates AUC from the imported HPLC matrix using chromatogram names.
Then it merges those AUC values back into the CCC metadata using `HPLC_filename`.

That is what allows the app to:
1. connect each chromatogram to the correct CCC tube
2. identify the `FI` and `FS` pair
3. calculate both:
   - `FS / FI`
   - `FI / FS`

### Safer design
This is safer than relying on an external metadata file, because the pairing logic
is already encoded by the metadata generated inside `ss_ccc`.
"""
            )
    processed_hplc_df = st.session_state.get("tab6_aligned_df", None)

    if processed_hplc_df is None or processed_hplc_df.empty:
        st.info("First load and process HPLC data in tab 6.")
    elif metadata_df is None or metadata_df.empty:
        st.info("Metadata must be available from tab 4.")
    else:
        st.markdown("### 1. Region definition")

        n_regions = st.number_input(
            "Number of RT regions",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            key="n_regions_tab7",
        )

        region_rows = []
        for i in range(int(n_regions)):
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                rt_start = st.number_input(
                    f"R{i+1} start",
                    value=float(processed_hplc_df["RT(min)"].min()),
                    step=0.1,
                    key=f"r{i+1}_start_tab7",
                )
            with c2:
                rt_end = st.number_input(
                    f"R{i+1} end",
                    value=float(processed_hplc_df["RT(min)"].min()) + 1.0 + i,
                    step=0.1,
                    key=f"r{i+1}_end_tab7",
                )
            with c3:
                label = st.text_input(
                    f"R{i+1} label",
                    value=f"Region_{i+1}",
                    key=f"r{i+1}_label_tab7",
                )

            region_rows.append(
                {
                    "region_id": f"R{i+1}",
                    "label": label,
                    "rt_start": min(rt_start, rt_end),
                    "rt_end": max(rt_start, rt_end),
                }
            )

        regions_df = pd.DataFrame(region_rows)

        with st.expander("Integration regions table", expanded=False):
            st.dataframe(regions_df, use_container_width=True)

        st.markdown("### 2. Region preview on chromatogram")

        plot_df = processed_hplc_df.melt(
            id_vars="RT(min)",
            var_name="Sample",
            value_name="Intensity"
        ).dropna(subset=["Intensity"])

        preview_mode = st.selectbox(
            "Preview mode",
            ["Overlay", "Stacked"],
            index=0,
            key="preview_mode_tab7",
        )

        if preview_mode == "Overlay":
            fig_preview = px.line(
                plot_df,
                x="RT(min)",
                y="Intensity",
                color="Sample",
                title="Chromatogram preview with integration regions"
            )
        else:
            samples_sorted = sorted([c for c in processed_hplc_df.columns if c != "RT(min)"])
            stack_step = st.number_input(
                "Preview stack offset",
                value=2.0,
                step=0.5,
                key="preview_stack_step_tab7",
            )
            offset_map = {s: i * stack_step for i, s in enumerate(samples_sorted)}
            plot_df["Intensity_offset"] = plot_df.apply(
                lambda r: r["Intensity"] + offset_map[r["Sample"]],
                axis=1
            )
            fig_preview = px.line(
                plot_df,
                x="RT(min)",
                y="Intensity_offset",
                color="Sample",
                title="Stacked chromatogram preview with integration regions"
            )

        fig_preview = add_region_overlays(fig_preview, region_rows)
        fig_preview.update_layout(xaxis_title="RT (min)", yaxis_title="Intensity")
        st.plotly_chart(fig_preview, use_container_width=True)

        st.markdown("### 3. AUC calculation")
        do_integrate = st.button("Calculate AUC", key="calculate_auc_tab7")

        if do_integrate:
            auc_df = integrate_regions_from_df(processed_hplc_df, region_rows)
            st.session_state["auc_tab7"] = auc_df

        auc_df = st.session_state.get("auc_tab7", None)

        if auc_df is not None and not auc_df.empty:
            with st.expander("AUC table", expanded=False):
                st.dataframe(auc_df, use_container_width=True)

            st.markdown("### 4. Metadata integration")

            metadata_editor_df = st.session_state.get("uploaded_metadata_tab6", metadata_df.copy()).copy()
            metadata_editor_df["HPLC_filename"] = metadata_editor_df["HPLC_filename"].fillna("")

            edited_meta = st.data_editor(
                metadata_editor_df,
                use_container_width=True,
                num_rows="fixed",
                key="metadata_editor_tab7",
            )

            if st.button("Merge AUC with metadata", key="merge_auc_metadata_tab7"):
                merged_auc_df = merge_auc_with_metadata(auc_df, edited_meta)
                st.session_state["merged_auc_tab7"] = merged_auc_df

            merged_auc_df = st.session_state.get("merged_auc_tab7", None)

            if merged_auc_df is not None and not merged_auc_df.empty:
                with st.expander("Merged metadata + AUC", expanded=False):
                    st.dataframe(merged_auc_df, use_container_width=True)

                st.markdown("### 5. Keq calculation")

                if st.button("Calculate Keq (FS/FI and FI/FS)", key="calc_keq_tab7"):
                    keq_df = calculate_keq_from_metadata(merged_auc_df)
                    st.session_state["keq_tab7"] = keq_df

                keq_df = st.session_state.get("keq_tab7", None)

                if keq_df is not None and not keq_df.empty:
                    with st.expander("Keq table", expanded=True):
                        st.dataframe(keq_df, use_container_width=True)

                    keq_plot_candidates = [c for c in keq_df.columns if c.endswith("_FS_over_FI")]
                    if keq_plot_candidates:
                        selected_keq_col = st.selectbox(
                            "Keq column to visualize",
                            options=keq_plot_candidates,
                            key="selected_keq_plot_col_tab7",
                        )

                        fig_keq = px.bar(
                            keq_df,
                            x="pair_id",
                            y=selected_keq_col,
                            color="sample_id",
                            title=f"Keq plot: {selected_keq_col}",
                            barmode="group",
                        )
                        fig_keq.update_layout(
                            xaxis_title="Pair ID",
                            yaxis_title="Keq"
                        )
                        st.plotly_chart(fig_keq, use_container_width=True)

                    st.markdown("### 6. Downloads")
                    st.download_button(
                        "Download regions table (CSV)",
                        data=convert_df_to_csv(regions_df),
                        file_name="integration_regions.csv",
                        mime="text/csv",
                    )

                    st.download_button(
                        "Download AUC table (CSV)",
                        data=convert_df_to_csv(auc_df),
                        file_name="auc_table.csv",
                        mime="text/csv",
                    )

                    st.download_button(
                        "Download merged metadata + AUC (CSV)",
                        data=convert_df_to_csv(merged_auc_df),
                        file_name="metadata_auc_merged.csv",
                        mime="text/csv",
                    )

                    st.download_button(
                        "Download Keq table (CSV)",
                        data=convert_df_to_csv(keq_df),
                        file_name="keq_table.csv",
                        mime="text/csv",
                    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Initial app structure for CCC solvent-system planning, metadata generation, and future OT-2 integration."
)

