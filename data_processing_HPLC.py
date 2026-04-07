# data_processing.py
# =============================================================================
# Imports & optional dependencies (organized)
# =============================================================================

# --- Standard Library --------------------------------------------------------
import os
import glob
import csv
from io import StringIO
from types import SimpleNamespace

# --- Core Data Handling ------------------------------------------------------
import numpy as np
import pandas as pd

# --- SciPy: signal processing & statistics ----------------------------------
from scipy.signal import correlate, find_peaks
from scipy.interpolate import interp1d
from scipy.stats import norm  # keep if you use the PDF/CDF; else remove
import scipy.stats as stats    # general stats namespace (ttest, pearsonr, etc.)

# --- Visualization (first-party always available) ---------------------------
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

# Matplotlib is lighter than seaborn and often present
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# --- Seaborn (optional; guarded) --------------------------------------------
try:
    import seaborn as sns
    HAVE_SEABORN = True
except Exception:
    HAVE_SEABORN = False

# --- Machine Learning (scikit-learn) — optional ------------------------------
HAVE_SKLEARN = True
try:
    from sklearn.decomposition import PCA as _SK_PCA
    from sklearn.cross_decomposition import PLSRegression as _SK_PLSRegression
    from sklearn.preprocessing import OneHotEncoder as _SK_OneHotEncoder
    from sklearn.model_selection import cross_val_score as _SK_cross_val_score, GroupKFold as _SK_GroupKFold
    from sklearn.metrics import r2_score as _SK_r2_score
except Exception:
    HAVE_SKLEARN = False
    _SK_PCA = _SK_PLSRegression = _SK_OneHotEncoder = None
    _SK_cross_val_score = _SK_GroupKFold = _SK_r2_score = None

def require_sklearn(feature: str = "this feature"):
    """Raise a clear error if scikit-learn is required but not installed."""
    if not HAVE_SKLEARN:
        raise ImportError(
            f"scikit-learn is required for {feature}. "
            "Install it with: pip install scikit-learn"
        )


# --- Clustering / HCA --------------------------------------------------------
from scipy.cluster.hierarchy import linkage, fcluster
import plotly.figure_factory as ff

# --- Tensor decompositions (optional; tensorly) ------------------------------
try:
    from tensorly.decomposition import parafac
    from tensorly import tensor
    HAVE_TENSORLY = True
except Exception:
    HAVE_TENSORLY = False


# =============================================================================
# Optional lib: pyicoshift (graceful fallback, robust wrapper)
# =============================================================================
import numpy as np

HAVE_PYICOSHIFT = False
ICOSHF_CALLABLE = None  # will become a callable(X, segments=<int>, reference=<1d array|int|None>) -> aligned X

# ---- call helper: try common kw aliases then positional ----
def _call_with_param_aliases(func_or_method, X, segments, reference):
    tries = [
        dict(segments=segments, reference=reference),
        dict(n_intervals=segments, reference=reference),
        dict(intervals=segments, reference=reference),
        dict(nSegments=segments, reference=reference),
        dict(segments=segments, target=reference),
        dict(n_intervals=segments, target=reference),
        dict(intervals=segments, target=reference),
        dict(nSegments=segments, target=reference),
    ]
    last_err = None
    for kwargs in tries:
        try:
            return func_or_method(X, **kwargs)
        except TypeError as e:
            last_err = e
            continue
    try:
        return func_or_method(X, segments, reference)
    except Exception as e:
        raise TypeError(
            f"ICOSHF backend rejected all calling conventions. Last error: {e}"
        ) from (last_err or e)

# ---- backend normalizer: class | instance | function -> unified callable ----
def _normalize_icoshift_backend(backend):
    try:
        obj = backend() if isinstance(backend, type) else backend
    except TypeError:
        obj = backend  # not instantiable → treat as bare function

    def _runner(X, segments=None, reference=None):
        # preference order
        if hasattr(obj, "align") and callable(getattr(obj, "align")):
            return _call_with_param_aliases(obj.align, X, segments, reference)
        if hasattr(obj, "fit_transform") and callable(getattr(obj, "fit_transform")):
            return _call_with_param_aliases(obj.fit_transform, X, segments, reference)
        if callable(obj):
            return _call_with_param_aliases(obj, X, segments, reference)
        raise TypeError("Unsupported ICOSHIFT backend: no align/fit_transform/callable interface.")
    return _runner

# ---- try to import pyicoshift in either form, but ALWAYS wrap via normalizer ----
try:
    from pyicoshift import icoshift as _ico_fn  # function API
    ICOSHF_CALLABLE = _normalize_icoshift_backend(_ico_fn)
    HAVE_PYICOSHIFT = True
except Exception:
    try:
        from pyicoshift import Icoshift as _IcoshiftCls  # class API
        ICOSHF_CALLABLE = _normalize_icoshift_backend(_IcoshiftCls)
        HAVE_PYICOSHIFT = True
    except Exception:
        HAVE_PYICOSHIFT = False
        ICOSHF_CALLABLE = None

# ---- public helper: safe alignment call ----
def align_samples_using_icoshift(df, n_intervals=20, target=None):
    """
    df: DataFrame with columns: RT(min), sample1, sample2, ...
    Returns DataFrame with same columns, aligned (RT unchanged; intensities shifted).
    """
    if df is None or df.shape[1] <= 1:
        raise ValueError("Need a matrix with at least one sample column.")
    if n_intervals is None or int(n_intervals) < 1:
        raise ValueError("n_intervals must be a positive integer.")
    if not HAVE_PYICOSHIFT or ICOSHF_CALLABLE is None:
        raise RuntimeError("Icoshift backend not available. Install 'pyicoshift' in this environment.")

    rt = df["RT(min)"].to_numpy(dtype=float)
    sample_cols = [c for c in df.columns if c != "RT(min)"]
    X = np.vstack([df[c].to_numpy(dtype=float) for c in sample_cols])

    # NaN-safe
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)

    # reference spectrum
    if isinstance(target, str):
        t = target.lower()
        if t == "median":
            ref = np.median(X, axis=0)
        elif t == "mean":
            ref = np.mean(X, axis=0)
        else:
            raise ValueError(f"Unknown target='{target}'. Use 'median', 'mean', an int index, or a 1D array.")
    elif isinstance(target, (int, np.integer)):
        if target < 0 or target >= X.shape[0]:
            raise IndexError(f"target index {target} out of range (0..{X.shape[0]-1}).")
        ref = X[int(target), :]
    elif target is None:
        ref = np.median(X, axis=0)
    else:
        ref = np.asarray(target, dtype=float)
        if ref.ndim != 1 or ref.shape[0] != X.shape[1]:
            raise ValueError("Provided reference must be 1D and match number of RT points.")

    # call backend (some libs return (aligned, shifts) → keep first)
    try:
        out = ICOSHF_CALLABLE(X, segments=int(n_intervals), reference=ref)
        X_aligned = out[0] if (isinstance(out, (tuple, list)) and len(out) > 0) else out
    except Exception as e:
        raise RuntimeError(
            f"Icoshift failed. X={X.shape}, segments={int(n_intervals)}, ref={ref.shape if hasattr(ref,'shape') else 'scalar'}; "
            f"backend={type(ICOSHF_CALLABLE)}; error={type(e).__name__}: {e}"
        ) from e

    if X_aligned.shape != X.shape:
        raise RuntimeError(f"Icoshift returned shape {X_aligned.shape}, expected {X.shape}.")

    out_df = df.copy()
    for i, c in enumerate(sample_cols):
        out_df[c] = X_aligned[i, :]
    return out_df



# =============================================================================
# Encoding detection (chardet / charset-normalizer / robust fallbacks)
# =============================================================================
HAVE_CHARDET = False
HAVE_CHARSET_NORMALIZER = False
try:
    import chardet  # pip install chardet
    HAVE_CHARDET = True
except Exception:
    try:
        from charset_normalizer import from_bytes as cn_from_bytes  # pip install charset-normalizer
        HAVE_CHARSET_NORMALIZER = True
    except Exception:
        pass

def sniff_encoding(data: bytes) -> str:
    """
    Detect file encoding from raw bytes.
    Preference order:
      1) chardet
      2) charset-normalizer
      3) common guesses ('utf-8', 'cp1252', 'latin1')
    Always returns a usable encoding (default 'latin1').
    """
    if HAVE_CHARDET:
        try:
            det = chardet.detect(data)
            enc = (det or {}).get("encoding") or ""
            if enc:
                return enc
        except Exception:
            pass

    if HAVE_CHARSET_NORMALIZER:
        try:
            res = cn_from_bytes(data)
            best = res.best()
            if best and best.encoding:
                return best.encoding
        except Exception:
            pass

    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            data.decode(enc)
            return enc
        except Exception:
            continue
    return "latin1"

# =============================================================================
# Public API hint (optional): list what you intend to export from this module
# =============================================================================
__all__ = [
    # flags
    "HAVE_MPL", "HAVE_SEABORN", "HAVE_TENSORLY", "HAVE_PYICOSHIFT",
    # helpers
    "sniff_encoding", "align_with_icoshift",
    # commonly used imports re-exposed (optional style)
    "np", "pd", "px", "go", "sp", "plt", "ff",
    "PCA", "PLSRegression", "OneHotEncoder", "cross_val_score",
    "GroupKFold", "r2_score", "linkage", "fcluster",
    "correlate", "find_peaks", "interp1d", "stats", "norm"
]

# --------------------------------------------------------------------------
#               1) DATA IMPORT
# --------------------------------------------------------------------------
def process_txt_files(input_folder):
    """
    Reads all .txt files in input_folder, extracts the table that starts at the line containing
    'R.Time (min)' and 'Intensity' (tab-separated), converts numeric decimals from comma to dot,
    and saves the table as a CSV file in a subfolder called 'data'.
    """
    # Create the output subfolder 'data' if it doesn't exist
    output_folder = os.path.join(input_folder, 'data')
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Find the index of the header line containing both "R.Time (min)" and "Intensity"
            header_index = None
            for idx, line in enumerate(lines):
                if "R.Time (min)" in line and "Intensity" in line:
                    header_index = idx
                    break
            
            if header_index is None:
                print(f"Header not found in {filename}. Skipping file.")
                continue
            
            # Extract the table data starting from the header row
            table_data = "".join(lines[header_index:])
            
            # Read the table into a DataFrame using StringIO, assuming tab-delimited data.
            # The parameter `decimal=','` converts numbers with commas as decimal separators.
            try:
                df = pd.read_csv(StringIO(table_data), sep='\t', decimal=',')
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
            
            # Construct the output CSV file path
            output_filename = os.path.splitext(filename)[0] + '.csv'
            output_path = os.path.join(output_folder, output_filename)
            
            # Save the DataFrame as a CSV file. Floats will be saved with the standard '.' as the decimal separator.
            df.to_csv(output_path, index=False)
            #print(f"Processed and saved: {output_path}")

def combine_csv_files(data_folder, output_csv=None):
    """
    Combines CSV files in the specified folder by taking the first column as the common axis
    and using the second column from each file as separate data columns.
    
    Parameters:
        data_folder (str): Path to the folder containing CSV files.
        output_csv (str, optional): If provided, the combined DataFrame will be saved to this CSV file.
    
    Returns:
        pd.DataFrame: The combined DataFrame.
    """
    combined_df = None
    
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the folder.")
        return None
    
    for file in csv_files:
        file_path = os.path.join(data_folder, file)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        
        # Keep only the first two columns
        if df.shape[1] < 2:
            print(f"File {file} has less than 2 columns. Skipping.")
            continue
        
        df = df.iloc[:, :2]
        
        # Rename columns:
        # Set a common name for the axis column and use the file name for the value column.
        base_name = os.path.splitext(file)[0]
        df.columns = ['RT(min)', base_name]
        
        # Merge the dataframes on the 'axis' column.
        if combined_df is None:
            combined_df = df
        else:
            # Use an outer join in case the axis values do not match exactly.
            combined_df = pd.merge(combined_df, df, on='RT(min)', how='outer')
    
    if combined_df is not None:
        # Optionally sort the combined DataFrame by the axis
        combined_df.sort_values(by='RT(min)', inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        
        if output_csv:
            combined_df.to_csv(output_csv, index=False)
            print(f"Combined CSV saved to {output_csv}")
    else:
        print("No data to combine.")
    
    return combined_df


def extract_rt_and_wavelength_fixed(input_file, target_wavelength):
    with open(input_file, 'r', encoding='latin1') as file:
        lines = file.readlines()

    header_idx = next((i for i, line in enumerate(lines) if "R.Time (min)" in line), None)
    if header_idx is None:
        raise ValueError("Header with 'R.Time (min)' not found.")

    for i in range(header_idx + 1, len(lines)):
        parts = lines[i].strip().split('\t')[1:]
        if all(p.strip().isdigit() for p in parts):
            wl_idx = i
            break
    else:
        raise ValueError("No numeric wavelength row found.")

    wavelengths = [int(w) / 100 for w in lines[wl_idx].strip().split('\t')[1:]]
    col_index = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - target_wavelength))
    selected_wavelength = round(wavelengths[col_index], 2)

    data = []
    for line in lines[wl_idx + 1:]:
        parts = line.strip().split('\t')
        if len(parts) <= col_index + 1:
            continue
        try:
            rt = float(parts[0].replace(',', '.'))
            intensity = float(parts[col_index + 1].replace(',', '.'))
            data.append((rt, intensity))
        except ValueError:
            continue

    df = pd.DataFrame(data, columns=['RT(min)', f'Intensity_{selected_wavelength}nm'])
    return df

def import_3D_data(input_folder, target_wavelength=254.0, output_filename='combined_wavelength_data.csv'):
    combined_df = None

    # Cria subpasta 'data'
    output_folder = os.path.join(input_folder, 'data')
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            try:
                df = extract_rt_and_wavelength_fixed(file_path, target_wavelength)
                df.rename(columns={df.columns[1]: os.path.splitext(filename)[0]}, inplace=True)
                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = pd.merge(combined_df, df, on='RT(min)', how='outer')
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

    if combined_df is not None:
        combined_df.sort_values(by='RT(min)', inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        output_path = os.path.join(output_folder, output_filename)
        combined_df.to_csv(output_path, index=False)
        print(f"✅ Combined file saved to: {output_path}")
    else:
        print("⚠️ No data combined.")

    return combined_df


### Remove ends          
"""def filter_rt_range(data, start_rt, end_rt, axis_column=None):
    ""
    Filters the DataFrame to keep only rows where the values in the axis column are between start_rt and end_rt.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        start_rt (float): The lower bound of the RT range.
        end_rt (float): The upper bound of the RT range.
        axis_column (str, optional): The column name representing the RT axis.
                                     If None, the function uses the first column.
    
    Returns:
        pd.DataFrame: The filtered DataFrame.
    ""
    # Use the provided axis_column or default to the first column
    if axis_column is None:
        axis_column = data.columns[0]
    
    filtered_data = data[(data[axis_column] >= start_rt) & (data[axis_column] <= end_rt)]
    return filtered_data
"""
def filter_rt_range(data: pd.DataFrame, start_rt: float, end_rt: float, axis_column: str | None = None) -> pd.DataFrame:
    """
    Keep only rows where the RT axis is between start_rt and end_rt (inclusive).
    If axis_column is None, the first DataFrame column is used.
    """
    if data is None or data.empty:
        return data
    if axis_column is None:
        axis_column = data.columns[0]
    lo = min(start_rt, end_rt)
    hi = max(start_rt, end_rt)
    mask = (data[axis_column] >= lo) & (data[axis_column] <= hi)
    return data.loc[mask].reset_index(drop=True)

def substitute_region_with_zeros(data, region_start, region_end, axis_column=None):
    """
    Sets values to zero for all rows where the value in the axis column is between region_start and region_end.
    
    Parameters:
        data (pd.DataFrame): The input DataFrame.
        region_start (float): The starting value of the region.
        region_end (float): The ending value of the region.
        axis_column (str, optional): The column name representing the axis (e.g., retention time).
                                     If None, the function uses the first column of the DataFrame.
    
    Returns:
        pd.DataFrame: A modified DataFrame with values in the specified region set to zero (for all columns except the axis).
    """
    # Use the provided axis_column or default to the first column
    if axis_column is None:
        axis_column = data.columns[0]
    
    # Create a copy of the DataFrame to avoid modifying the original data
    modified_data = data.copy()
    
    # Create a boolean mask for the region to be substituted with zeros
    mask = (modified_data[axis_column] >= region_start) & (modified_data[axis_column] <= region_end)
    
    # Apply zeros to all columns except the axis column for rows where the mask is True
    for col in modified_data.columns:
        if col != axis_column:
            modified_data.loc[mask, col] = 0
            
    return modified_data


# --------------------------------------------------------------------------
#               2) DATA PLOTING
# --------------------------------------------------------------------------
def create_chromatogram_plot(dataframe, 
                             x_axis_col='RT(min)', 
                             start_column=1, 
                             end_column=25, 
                             title='Chromatogram Overlapping',
                             xaxis_title='RT (min)',
                             yaxis_title='Intensity',
                             legend_title='Samples',
                             output_dir='images', 
                             output_file='chromatogram_overlapping.html',
                             show_fig=False):
    """
    Creates a Plotly figure of overlapping chromatograms from the given DataFrame,
    using the specified range of columns (by index) for the sample traces.
    
    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data. It must include the x-axis column.
        x_axis_col (str): Name of the column to use for the x-axis. Default is 'RT(min)'.
        start_column (int): Index of the first column to plot (after the x-axis column). Default is 1.
        end_column (int): Index of the last column to plot (inclusive). Default is 25.
        title (str): Title of the plot.
        xaxis_title (str): Label for the x-axis.
        yaxis_title (str): Label for the y-axis.
        legend_title (str): Title for the legend.
        output_dir (str): Directory where the HTML file will be saved.
        output_file (str): Filename for the output HTML file.
        
    Returns:
        None. The function saves the Plotly figure as an HTML file.
    """
    fig = go.Figure()

    # Iterate over the specified range of columns and add each as a trace.
    for column in dataframe.columns[start_column:end_column + 1]:
        fig.add_trace(go.Scatter(x=dataframe[x_axis_col],
                                 y=dataframe[column],
                                 mode='lines',
                                 name=column))
    
    # Update the figure layout.
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title=legend_title,
        hovermode='closest'
    )
    
    # Create the output directory if it doesn't exist and save the HTML file.
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    fig.write_html(output_path)
    print(f"Plot saved as: {output_path}")
    
    # Optionally display the figure
    if show_fig:
        fig.show()

    
def create_stacked_chromatogram_plot(dataframe, 
                                     x_axis_col='RT(min)', 
                                     start_column=1, 
                                     end_column=25, 
                                     gap=10000, 
                                     title='Stacked Chromatograms with Larger Gaps',
                                     xaxis_title='RT (min)',
                                     yaxis_title='Intensity (Stacked)',
                                     legend_title='Samples',
                                     output_dir='images', 
                                     output_file='stacked_chromatograms_larger_gap.html',
                                     show_fig=False):
    """
    Creates a stacked chromatogram Plotly figure by offsetting each trace vertically by a specified gap.
    
    Parameters:
        dataframe (pd.DataFrame): DataFrame containing the chromatogram data. Must include the x-axis column.
        x_axis_col (str): Name of the column to use for the x-axis (default 'RT(min)').
        start_column (int): Index of the first sample column to plot (default 1).
        end_column (int): Index of the last sample column to plot (inclusive, default 25).
        gap (float): Vertical offset between traces. Set to 0 for no offset (default 10000).
        title (str): Title of the figure.
        xaxis_title (str): Label for the x-axis.
        yaxis_title (str): Label for the y-axis.
        legend_title (str): Title for the legend.
        output_dir (str): Directory where the HTML file will be saved.
        output_file (str): Filename for the output HTML file.
        show_fig (bool): If True, the figure will be displayed after saving (default False).
        
    Returns:
        None. The function saves the Plotly figure as an HTML file.
    """
    fig = go.Figure()
    
    plotted_columns = 0  # Track how many valid traces are plotted
    
    # Iterate over the specified sample columns, offsetting each trace by gap * index.
    for i, column in enumerate(dataframe.columns[start_column:end_column + 1]):
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            fig.add_trace(go.Scatter(
                x=dataframe[x_axis_col],
                y=dataframe[column] + plotted_columns * gap,
                mode='lines',
                name=column
            ))
            plotted_columns += 1  # Increment only if we plotted
        else:
            print(f"⚠️ Skipping non-numeric column: {column}")

    if plotted_columns == 0:
        print("❌ No numeric columns found to plot. Check your dataframe.")
        return
    
    # Update the figure layout
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title=legend_title,
        hovermode='closest'
    )
    
    # Create the output directory if it doesn't exist and save the figure as an HTML file.
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    fig.write_html(output_path)
    print(f"✅ Plot saved as: {output_path}")
    
    # Optionally display the figure
    if show_fig:
        fig.show()

def analyze_and_visualize_peaks(dataframe, 
                                x_axis_col='RT(min)', 
                                start_column=1, 
                                end_column=25, 
                                peak_height=5000, 
                                peaks_csv_path='images/peaks_count.csv', 
                                visualization_dir='images/peaks_visualization',
                                show_fig=False):
    """
    Detects peaks in the specified sample columns of a DataFrame, saves a CSV file with the peak counts,
    and creates HTML visualizations for each sample showing the chromatogram with detected peaks.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing the chromatogram data. Must include the x-axis column.
        x_axis_col (str): Name of the column to use for the x-axis (default 'RT(min)').
        start_column (int): Index of the first sample column to process (default 1).
        end_column (int): Index of the last sample column to process (inclusive, default 25).
        peak_height (float): Minimum height threshold for peak detection (default 5000).
        peaks_csv_path (str): File path to save the CSV with peak counts.
        visualization_dir (str): Directory where individual peak visualization HTML files will be saved.
        show_fig (bool): If True, each figure will be displayed after creation (default False).

    Returns:
        pd.DataFrame: A DataFrame containing the number of detected peaks per sample.
    """
    # Ensure the output directories exist
    os.makedirs(os.path.dirname(peaks_csv_path), exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    # ----------------------------
    # Part 1: Count Peaks & Save CSV
    # ----------------------------
    peaks_count = {}
    
    for column in dataframe.columns[start_column:end_column + 1]:
        # Extract intensity values from the current sample column
        intensity_values = dataframe[column].values
        
        # Detect peaks in the intensity values using the specified height threshold
        peaks, _ = find_peaks(intensity_values, height=peak_height)
        
        # Store the number of detected peaks for this sample
        peaks_count[column] = len(peaks)
    
    # Convert the dictionary to a DataFrame and save as CSV
    peaks_count_df = pd.DataFrame(list(peaks_count.items()), columns=['Sample', 'Number of Peaks'])
    peaks_count_df.to_csv(peaks_csv_path, index=False)
    print(f"Peak counts saved to: {peaks_csv_path}")
    
    # ----------------------------
    # Part 2: Visualize Peaks for Each Sample
    # ----------------------------
    for column in dataframe.columns[start_column:end_column + 1]:
        # Extract intensity and x-axis (RT) values for the current sample
        intensity_values = dataframe[column].values
        rt_values = dataframe[x_axis_col].values

        # Detect peaks for the current sample
        peaks, _ = find_peaks(intensity_values, height=peak_height)
        
        # Create a Plotly figure
        fig = go.Figure()

        # Add the chromatogram trace
        fig.add_trace(go.Scatter(
            x=rt_values,
            y=intensity_values,
            mode='lines',
            name=column,
            line=dict(color='blue')
        ))

        # Add markers for the detected peaks
        fig.add_trace(go.Scatter(
            x=rt_values[peaks],
            y=intensity_values[peaks],
            mode='markers',
            name='Detected Peaks',
            marker=dict(color='red', size=8, symbol='circle-open')
        ))

        # Update the layout of the figure
        fig.update_layout(
            title=f'Chromatogram with Detected Peaks for {column}',
            xaxis_title=x_axis_col,
            yaxis_title='Intensity',
            legend_title='Legend',
            hovermode='closest'
        )

        # Save the figure as an HTML file in the visualization directory
        output_file = os.path.join(visualization_dir, f'{column}_peaks.html')
        fig.write_html(output_file)
        print(f"Peak visualization saved for '{column}' to: {output_file}")

        # Optionally display the figure
        if show_fig:
            fig.show()
    
    return peaks_count_df


def create_fraction_chromatogram_plot(dataframe, 
                                      selected_rt, 
                                      x_axis_col='RT(min)', 
                                      output_dir='images', 
                                      show_fig=False):
    """
    Creates a line plot of intensities across samples (fractions) at the retention time (RT) closest to a specified value.
    
    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing RT values and sample intensities.
                                  The first column (default) should be the RT values (e.g., 'RT(min)'),
                                  and subsequent columns represent different samples.
        selected_rt (float): The desired retention time to plot.
        x_axis_col (str): The name of the RT column in the DataFrame. Default is 'RT(min)'.
        output_dir (str): Directory where the HTML file will be saved. Default is 'images'.
        show_fig (bool): If True, the function will display the figure after creation. Default is False.
    
    Returns:
        None. The function saves the plot as an HTML file.
    """
    # Find the closest retention time to the selected value
    closest_rt = dataframe[x_axis_col].iloc[(dataframe[x_axis_col] - selected_rt).abs().argsort()[:1]].values[0]
    
    # Find the row corresponding to the closest retention time
    selected_row = dataframe[dataframe[x_axis_col] == closest_rt]
    
    # Extract the sample names and the corresponding intensities from the selected row.
    # Assumes the first column is RT and all other columns are samples.
    sample_names = dataframe.columns[1:]
    intensities = selected_row.iloc[0, 1:].values

    # Create a line plot of intensities across samples
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample_names,
        y=intensities,
        mode='lines+markers',
        name=f'RT = {closest_rt} min'
    ))
    
    # Update the layout of the figure
    fig.update_layout(
        title=f'Intensity Across Samples at Closest RT = {closest_rt} min to Selected RT = {selected_rt} min',
        xaxis_title='Samples',
        yaxis_title='Intensity',
        legend_title='Retention Time',
        hovermode='closest'
    )
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'intensity_across_samples_{selected_rt}min.html')
    
    # Save the figure as an HTML file
    fig.write_html(output_file)
    print(f"Plot saved as: {output_file}")
    
    # Optionally, display the figure
    if show_fig:
        fig.show()

def create_vertical_multiplot(dataframes, titles,
                              x_axis_col='RT(min)', 
                              start_column=1, end_column=None, 
                              xaxis_title='RT(min)', 
                              yaxis_title='Intensity', 
                              legend_title='Samples',
                              output_dir='images', 
                              output_file='aligned_LC_multiplot.html',
                              show_fig=True):
    n = len(dataframes)
    fig = sp.make_subplots(rows=n, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=titles)
    for i, df in enumerate(dataframes, start=1):
        x = df[x_axis_col].values
        if end_column is None:
            intensity_cols = df.columns[start_column:]
        else:
            intensity_cols = df.columns[start_column:end_column+1]
        for col in intensity_cols:
            fig.add_trace(
                go.Scattergl(x=x, y=df[col].values, mode='lines', name=col),
                row=i, col=1
            )
        fig.update_yaxes(title_text=yaxis_title, row=i, col=1)
    fig.update_layout(
        title="Comparison of Aligned LC Data",
        xaxis_title=xaxis_title,
        legend_title=legend_title,
        height=500 * n,
        margin=dict(t=100)
    )
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, output_file))
    if show_fig:
        fig.show()
    return fig

def plot_histogram_with_distribution(data, output_dir='images', file_name='histogram_with_distribution_curve.html', log_scale=False, x_range=None):
    melted_df = data.melt(value_name="Normalized Value")
    os.makedirs(output_dir, exist_ok=True)
    mean_val = melted_df["Normalized Value"].mean()
    std_val = melted_df["Normalized Value"].std()
    x_values = np.linspace(melted_df["Normalized Value"].min(), melted_df["Normalized Value"].max(), 100)
    normal_curve = norm.pdf(x_values, mean_val, std_val)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=melted_df["Normalized Value"], 
        histnorm='probability density', 
        opacity=0.6,
        name='Normalized Data'
    ))
    fig.add_trace(go.Scatter(
        x=x_values,
        y=normal_curve,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Normal Distribution Curve'
    ))
    fig.update_layout(
        title='Histogram with Normal Distribution Curve',
        xaxis_title='Normalized Value',
        yaxis_title='Density',
        showlegend=True
    )
    if log_scale:
        fig.update_xaxes(type="log")
    if x_range:
        fig.update_xaxes(range=x_range)
    fig.write_html(os.path.join(output_dir, file_name))
    return fig


def _ensure_numeric_df(df, exclude=("RT(min)",)):
    cols = [c for c in df.columns if c not in exclude]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return df[num].copy()

def plot_corr_heatmap(df: pd.DataFrame, method: str = "pearson", title: str = "Correlation heatmap"):
    dat = _ensure_numeric_df(df)
    corr = dat.corr(method=method)

    if HAVE_SEABORN:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        sns.heatmap(corr, cmap="viridis", annot=False, ax=ax)
        ax.set_title(title)
        return fig
    else:
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                coloraxis="coloraxis"
            )
        )
        fig.update_layout(title=title, coloraxis_colorscale="Viridis", xaxis_nticks=20)
        return fig  

def plot_pairgrid(df: pd.DataFrame, hue: str | None = None, title: str = "Pairwise plot"):
    dat = df.copy()
    if hue and hue in dat.columns:
        color_col = hue
    else:
        color_col = None

    num = [c for c in dat.columns if pd.api.types.is_numeric_dtype(dat[c])]
    dat_num = dat[num + ([color_col] if color_col else [])]

    if HAVE_SEABORN:
        import matplotlib.pyplot as plt
        g = sns.pairplot(dat_num, hue=color_col, corner=True, diag_kind="hist")
        g.fig.suptitle(title, y=1.02)
        return g.fig
    else:
        fig = px.scatter_matrix(dat_num, dimensions=num, color=color_col, title=title)
        return fig

def plot_box(df: pd.DataFrame, group_col: str | None = None, title: str = "Box plots"):
    """Boxplot by column"""
    dat = _ensure_numeric_df(df)
    if HAVE_SEABORN:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        if group_col and group_col in df.columns:
            
            m = pd.melt(df[[group_col] + dat.columns.tolist()], id_vars=group_col, var_name="Feature", value_name="Value")
            sns.boxplot(data=m, x="Feature", y="Value", hue=group_col, ax=ax)
        else:
            sns.boxplot(data=dat, ax=ax)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=90)
        return fig
    else:
        if group_col and group_col in df.columns:
            m = pd.melt(df[[group_col] + dat.columns.tolist()], id_vars=group_col, var_name="Feature", value_name="Value")
            fig = px.box(m, x="Feature", y="Value", color=group_col, title=title)
        else:
            m = pd.melt(dat, var_name="Feature", value_name="Value")
            fig = px.box(m, x="Feature", y="Value", title=title)
        fig.update_layout(xaxis_tickangle=90)
        return fig

def compare_normalization_plots(
    before_df, 
    after_df, 
    sample_limit=50, 
    exclude_columns=None, 
    title_before='Before Normalization',
    title_after='After Normalization',
    show_full_xrange=False,
    zoom_in_std=None
):
    """
    Creates a 2x2 grid of plots:
      - Top-left: Density plot of all values before normalization
      - Top-right: Density plot of all values after normalization 
                   (range controlled by show_full_xrange or zoom_in_std).
      - Bottom-left: Box plot (at most sample_limit columns) before normalization
      - Bottom-right: Box plot (at most sample_limit columns) after normalization

    Parameters
    ----------
    before_df : pd.DataFrame
        DataFrame containing the data prior to normalization.
    after_df : pd.DataFrame
        DataFrame containing the data after normalization.
    sample_limit : int
        Maximum number of columns to display in the box plots.
    exclude_columns : list or None
        List of columns to exclude from both DataFrames (e.g. axis columns like "RT (min)").
    title_before : str
        Title for the "before" plots.
    title_after : str
        Title for the "after" plots.
    show_full_xrange : bool
        If True, the top-right density plot will show the entire range of values.
        If False, it will restrict the view to the 1st and 99th percentiles (unless zoom_in_std is used).
    zoom_in_std : float or None
        If not None, the top-right density plot will show mean ± (zoom_in_std * standard deviation).
        This overrides the percentile-based or full-range display.
    """
    # Copy DataFrames so we don't modify the originals
    df_before = before_df.copy()
    df_after = after_df.copy()
    
    # Exclude certain columns if requested
    if exclude_columns is not None:
        df_before.drop(columns=exclude_columns, errors='ignore', inplace=True)
        df_after.drop(columns=exclude_columns, errors='ignore', inplace=True)
    
    # Flatten the entire DataFrame values for density plots
    all_values_before = df_before.values.flatten()
    all_values_after = df_after.values.flatten()
    
    # Calculate some statistics for controlling the x-axis range
    lower_limit = np.percentile(all_values_after, 1)
    upper_limit = np.percentile(all_values_after, 99)
    mean_val = np.mean(all_values_after)
    std_val = np.std(all_values_after)

    # Limit columns for box plot (for readability)
    limited_cols_before = df_before.columns[:sample_limit]
    limited_cols_after = df_after.columns[:sample_limit]
    
    # Create a 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    # --- Top-left: Density of all values BEFORE normalization ---
    sns.kdeplot(x=all_values_before, ax=axes[0,0], color='steelblue', fill=True)
    axes[0,0].set_title(title_before, fontsize=12)
    axes[0,0].set_xlabel("Value")
    axes[0,0].set_ylabel("Density")
    
    # --- Top-right: Density of all values AFTER normalization ---
    sns.kdeplot(x=all_values_after, ax=axes[0,1], color='darkorange', fill=True)
    axes[0,1].set_title(title_after, fontsize=12)
    axes[0,1].set_xlabel("Value")
    axes[0,1].set_ylabel("Density")
    
    # Control the x-axis range for the "after" density plot
    if zoom_in_std is not None:
        # Show mean ± zoom_in_std * std_val
        axes[0,1].set_xlim(mean_val - zoom_in_std * std_val, 
                           mean_val + zoom_in_std * std_val)
    else:
        # If zoom_in_std is not used, check if we should show full range or restrict to 1st-99th percentiles
        if not show_full_xrange:
            axes[0,1].set_xlim(lower_limit, upper_limit)
        # else: do nothing => show the full range automatically

    # --- Bottom-left: Box plot BEFORE normalization ---
    sns.boxplot(data=df_before[limited_cols_before], ax=axes[1,0], color='steelblue')
    axes[1,0].tick_params(axis='x', rotation=90)
    axes[1,0].set_title(title_before, fontsize=12)
    axes[1,0].set_ylabel("Value")
    
    # --- Bottom-right: Box plot AFTER normalization ---
    sns.boxplot(data=df_after[limited_cols_after], ax=axes[1,1], color='darkorange')
    axes[1,1].tick_params(axis='x', rotation=90)
    axes[1,1].set_title(title_after, fontsize=12)
    axes[1,1].set_ylabel("Value")
    
    # Show the final figure
    plt.show()


def integrate_area_html_all(aligned_df, peaks, output_folder="images", base_filename="integration_plot"):
    """
    Computes the area under the curve (AUC) for each sample in the given DataFrame within
    specified retention time (RT) intervals defined in the peaks list. For each sample, an
    interactive Plotly HTML plot is created that highlights the integration regions.
    
    Parameters
    ----------
    aligned_df : pd.DataFrame
        DataFrame where the first column is "RT (min)" (retention time) and the remaining columns 
        are the intensities for each sample.
    peaks : list
        List of retention time boundaries in the form 
            [lower_rt1, upper_rt1, lower_rt2, upper_rt2, ...].
        The length of peaks must be even.
    output_folder : str, optional
        Folder (inside the "images" directory) in which to save the HTML plots (default "html_images").
    base_filename : str, optional
        Base name for the output files. Each file will be named as 
        "{base_filename}_{sample_name}.html" (default "integration_plot").
    
    Returns
    -------
    auc_df : pd.DataFrame
        A DataFrame where rows are samples and columns are each integration region 
        (e.g., "Region 1", "Region 2", ...) containing the corresponding AUC values.
    """
    # Check that peaks has an even number of elements.
    if len(peaks) % 2 != 0:
        raise ValueError("The peaks list must contain an even number of elements (pairs of lower and upper limits).")
    
    # Create the base output directory and subfolder for HTML plots.
    base_output_dir = "images"
    html_output_dir = os.path.join(base_output_dir, output_folder)
    os.makedirs(html_output_dir, exist_ok=True)
    
    # Extract retention time (RT) from the first column and convert to numeric values
    rt = pd.to_numeric(aligned_df.iloc[:, 0], errors='coerce').values
    
    sample_names = aligned_df.columns[1:]
    num_regions = len(peaks) // 2
    
    # Prepare a dictionary to collect AUC values per sample and per region.
    auc_data = {f"Region {i+1}": [] for i in range(num_regions)}
    auc_data["Sample"] = []
    
    # Loop over each sample to compute integration and create a plot
    for sample in sample_names:
        intensities = pd.to_numeric(aligned_df[sample], errors='coerce').values
        
        # For storing region-specific AUC values for this sample.
        sample_auc = []
        
        # Create the Plotly figure for this sample
        fig = go.Figure()
        
        # Add the full spectrum trace for the sample
        fig.add_trace(go.Scatter(
            x=rt,
            y=intensities,
            mode='lines',
            name=sample,
            line=dict(color='black')
        ))
        
        # Loop through each integration region (pair of limits)
        for i in range(0, len(peaks), 2):
            lower_rt = peaks[i]
            upper_rt = peaks[i+1]
            mask = (rt >= lower_rt) & (rt <= upper_rt)
            if np.sum(mask) >= 2:
                auc = np.trapz(intensities[mask], rt[mask])
            else:
                auc = 0  # or np.nan if preferred
            sample_auc.append(auc)
            
            # Add trace for the integration region
            fig.add_trace(go.Scatter(
                x=rt[mask],
                y=intensities[mask],
                mode='lines',
                name=f"Region {i//2 + 1}",
                line=dict(color='skyblue'),
                fill='tozeroy',
                opacity=0.5
            ))
            
            # Add vertical dashed lines for region boundaries
            fig.add_shape(
                type="line",
                x0=lower_rt,
                x1=lower_rt,
                y0=min(intensities),
                y1=max(intensities),
                line=dict(color="red", dash="dash")
            )
            fig.add_shape(
                type="line",
                x0=upper_rt,
                x1=upper_rt,
                y0=min(intensities),
                y1=max(intensities),
                line=dict(color="red", dash="dash")
            )
        
        # Store the sample name and its region-specific AUC values
        auc_data["Sample"].append(sample)
        for idx, auc_val in enumerate(sample_auc):
            auc_data[f"Region {idx+1}"].append(auc_val)
        
        # Update layout for the current sample plot
        fig.update_layout(
            title=f"Integration Regions for Sample: {sample}",
            xaxis_title="RT (min)",
            yaxis_title="Intensity",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Save the figure as an interactive HTML file in the designated folder
        output_filename = f"{base_filename}_{sample}.html"
        output_path = os.path.join(html_output_dir, output_filename)
        fig.write_html(output_path)
        print(f"Integration plot for sample '{sample}' saved as: {output_path}")
    
    # Create a DataFrame from the auc_data dictionary
    auc_df = pd.DataFrame(auc_data).set_index("Sample")
    
    return auc_df


def plot_overlayed_chromatograms(aligned_df, peaks, output_filename="overlayed_chromatograms.html", output_folder="images"):
    """
    Creates an interactive Plotly HTML plot that overlays the chromatograms for all samples 
    (with retention time in the first column and intensities in subsequent columns) and 
    highlights the integration regions defined by the peaks list.
    
    Parameters
    ----------
    aligned_df : pd.DataFrame
        DataFrame where the first column is "RT (min)" (retention time) and the remaining columns 
        are the intensities for each sample.
    peaks : list
        List of retention time boundaries in the form 
            [lower_rt1, upper_rt1, lower_rt2, upper_rt2, ...].
        The length of peaks must be even.
    output_filename : str, optional
        Name of the output HTML file (default "overlayed_chromatograms.html").
    output_folder : str, optional
        Folder (inside the "images" directory) in which to save the HTML plot (default "html_images_overlay").
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The interactive Plotly figure with overlayed chromatograms and integrated regions.
    """
    # Check that peaks has an even number of elements.
    if len(peaks) % 2 != 0:
        raise ValueError("The peaks list must contain an even number of elements (pairs of lower and upper limits).")
    
    # Create the output directory for HTML plots
#    base_output_dir = "images"
    html_output_dir = os.path.join(output_folder)
    os.makedirs(html_output_dir, exist_ok=True)
    
    # Extract retention time (RT) from the first column and convert to numeric values
    rt = pd.to_numeric(aligned_df.iloc[:, 0], errors='coerce').values
    
    # Initialize the Plotly figure
    fig = go.Figure()
    
    # Overlay each sample's chromatogram
    sample_names = aligned_df.columns[1:]
    for sample in sample_names:
        intensities = pd.to_numeric(aligned_df[sample], errors='coerce').values
        fig.add_trace(go.Scatter(
            x=rt,
            y=intensities,
            mode='lines',
            name=sample,
            line=dict(width=1),
            opacity=0.7
        ))
    
    # Add shapes (shaded regions and vertical lines) for each integration region
    num_regions = len(peaks) // 2
    for i in range(num_regions):
        lower_rt = peaks[2*i]
        upper_rt = peaks[2*i + 1]
        # Add a shaded region
        fig.add_vrect(
            x0=lower_rt, x1=upper_rt,
            fillcolor="skyblue", opacity=0.3, line_width=0,
            annotation_text=f"Region {i+1}", annotation_position="top left"
        )
        # Optionally, add vertical dashed lines at the boundaries
        fig.add_shape(
            type="line",
            x0=lower_rt, x1=lower_rt,
            y0=min(rt), y1=max(rt),
            line=dict(color="red", dash="dash")
        )
        fig.add_shape(
            type="line",
            x0=upper_rt, x1=upper_rt,
            y0=min(rt), y1=max(rt),
            line=dict(color="red", dash="dash")
        )
    
    # Update layout settings for a clear presentation
    fig.update_layout(
        title="Overlayed Chromatograms with Integration Regions",
        xaxis_title="RT (min)",
        yaxis_title="Intensity",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Save the interactive plot as an HTML file in the designated folder
    output_path = os.path.join(html_output_dir, output_filename)
    fig.write_html(output_path)
    print(f"Overlayed chromatograms plot saved as: {output_path}")
    
    return fig

def plot_boxplots_by_class(
    auc_df, 
    df_metadata, 
    classification_column='ATTRIBUTE_classification',
    output_folder='plots', 
    filename='boxplot_auc_by_class.png',
    show_plot=True
):
    """
    Plots and saves boxplots for each integration region in the AUC DataFrame, grouped by sample classification.
    
    Parameters:
    - auc_df: DataFrame with index as sample names and columns as region names (e.g., Region 1, Region 2).
    - df_metadata: DataFrame with metadata including sample classifications.
    - classification_column: Column name in df_metadata used for grouping samples.
    - output_folder: Folder to save the plot.
    - filename: Filename for the saved plot.
    - show_plot: If True, displays the plot inline.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, filename)

    # Prepare data
    auc_df = auc_df.copy()
    auc_df['Samples'] = auc_df.index
    merged_df = pd.merge(auc_df, df_metadata[['Samples', classification_column]], on='Samples')
    melted_df = merged_df.melt(id_vars=[classification_column], 
                               value_vars=[col for col in auc_df.columns if col != 'Samples'],
                               var_name='Region', value_name='AUC')

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Region', y='AUC', hue=classification_column, data=melted_df)
    plt.title('AUC by Region grouped by Sample Classification')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()


def import_uv_matrix(input_file, wavelength_min=210, wavelength_max=400, verbose=False):
    """
    Imports the full UV matrix from a PDA 3D .txt exported by Shimadzu software.
    Filters the wavelengths to a specified range (default 210–400 nm).

    Parameters:
        input_file (str): Path to the text file.
        wavelength_min (float): Minimum wavelength to keep (default 210 nm).
        wavelength_max (float): Maximum wavelength to keep (default 400 nm).
        verbose (bool): Print warnings and info.

    Returns:
        - rt_values: array of retention times (RT)
        - wavelengths: array of wavelengths (nm) AFTER filtering
        - matrix_df: DataFrame (rows = RT, columns = wavelengths)
    """
    with open(input_file, 'r', encoding='latin1') as file:
        lines = file.readlines()

    # Find index of the line with 'R.Time (min)'
    header_idx = next((i for i, line in enumerate(lines) if "R.Time (min)" in line), None)
    if header_idx is None:
        raise ValueError("Header with 'R.Time (min)' not found.")

    # Find the wavelength header line
    for i in range(header_idx + 1, len(lines)):
        parts = lines[i].strip().split('\t')[1:]
        if all(p.strip().isdigit() for p in parts):
            wl_idx = i
            break
    else:
        raise ValueError("No numeric wavelength row found after header.")

    # Get wavelengths (convert to float nm)
    wavelengths_all = [int(w) / 100 for w in lines[wl_idx].strip().split('\t')[1:]]

    # Figure out which columns to keep
    keep_indices = [i for i, wl in enumerate(wavelengths_all) if wavelength_min <= wl <= wavelength_max]
    filtered_wavelengths = [wavelengths_all[i] for i in keep_indices]

    if verbose:
        print(f"Original wavelengths: {len(wavelengths_all)} → Filtered to {len(filtered_wavelengths)} between {wavelength_min}-{wavelength_max} nm.")

    # Read the matrix (starting from wl_idx + 1)
    rt_values = []
    data_matrix = []
    for line_num, line in enumerate(lines[wl_idx + 1:], start=wl_idx + 2):
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue  # skip empty lines
        try:
            rt = float(parts[0].replace(',', '.'))
            intensities_raw = parts[1:]
            if len(intensities_raw) < len(wavelengths_all):
                if verbose:
                    print(f"⚠️ Skipping incomplete line {line_num}: {len(intensities_raw)} values, expected {len(wavelengths_all)}")
                continue  # skip incomplete lines
            if len(intensities_raw) > len(wavelengths_all):
                if verbose:
                    print(f"⚠️ Line {line_num}: Extra columns detected ({len(intensities_raw)} values). Truncating to {len(wavelengths_all)}.")
            # Truncate if too many columns, then filter the desired range
            full_intensities = [float(x.replace(',', '.')) for x in intensities_raw[:len(wavelengths_all)]]
            filtered_intensities = [full_intensities[i] for i in keep_indices]
            rt_values.append(rt)
            data_matrix.append(filtered_intensities)
        except ValueError:
            if verbose:
                print(f"⚠️ Skipping invalid numeric data at line {line_num}.")
            continue

    # Create DataFrame: rows = RT(min), columns = wavelengths (nm)
    matrix_df = pd.DataFrame(data_matrix, columns=[f"{wl:.2f}nm" for wl in filtered_wavelengths])
    matrix_df.insert(0, 'RT(min)', rt_values)

    return rt_values, filtered_wavelengths, matrix_df


def view_uv_spectrum_at_rt_interactive(matrix_df, target_rt, output_html="uv_spectrum.html", tolerance=0.05):
    """
    Selects the closest RT to the target_rt and saves an interactive UV spectrum plot as HTML.
    
    Parameters:
        matrix_df (DataFrame): DataFrame containing the UV matrix (RT(min) + wavelengths as columns).
        target_rt (float): The retention time to find the closest match.
        output_html (str): Filename for the saved HTML plot.
        tolerance (float): Max allowed difference between found RT and target_rt. Prints warning if larger.

    Returns:
        wavelengths (list): List of wavelengths (floats).
        intensities (list): List of intensities for the closest RT.
        rt_found (float): The RT(min) that was actually used.
    """
    # Find the closest row to the target_rt
    closest_idx = (matrix_df['RT(min)'] - target_rt).abs().idxmin()
    row = matrix_df.iloc[closest_idx]
    rt_found = row['RT(min)']
    
    # Warn if too far from target
    if abs(rt_found - target_rt) > tolerance:
        print(f"⚠️ Closest RT is {rt_found:.4f} min (difference: {abs(rt_found - target_rt):.4f} min)")
    
    # Extract wavelengths and intensities
    wavelengths = [float(col.replace('nm', '')) for col in matrix_df.columns if 'nm' in col]
    intensities = row[1:].values  # skip the 'RT(min)' column

    # Create interactive plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wavelengths,
        y=intensities,
        mode='lines+markers',
        hovertemplate='Wavelength: %{x} nm<br>Intensity: %{y:.2f}<extra></extra>',
        name=f'Spectrum at {rt_found:.2f} min'
    ))

    fig.update_layout(
        title=f"UV Spectrum at RT = {rt_found:.2f} min",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity",
        hovermode="x unified"
    )

    # Save as HTML
    fig.write_html(output_html)
    print(f"✅ Interactive plot saved as: {output_html}")

    return wavelengths, intensities, rt_found


def plot_uv_contour(
    matrix_df,
    output_html="uv_contour_plot.html",
    intensity_transform="none"  # Options: 'none', 'log', 'minmax', 'normalize', ('custom', func)
):
    """
    Plots a contour plot of the UV matrix and saves as an interactive HTML file.

    Parameters:
    - matrix_df: DataFrame with 'RT(min)' as first column and wavelengths as the other columns.
    - output_html: Filename for the saved HTML plot.
    - intensity_transform: How to process intensities before plotting.
        Options:
            'none'      = raw data
            'log'       = np.log1p(intensity)
            'minmax'    = (X - X.min()) / (X.max() - X.min())
            'normalize' = each wavelength normalized to its max
            ('custom', func) = a tuple with a custom function applied elementwise
    """
    # Extract axes and data
    rt_values = matrix_df.iloc[:, 0].values
    wavelengths = [float(col.replace("nm", "")) for col in matrix_df.columns[1:]]
    intensity_matrix = matrix_df.iloc[:, 1:].values.T  # Transpose (wavelengths x rt)

    # Apply intensity transformation
    if intensity_transform == "log":
        print("⚙️ Applying log transformation (log1p)...")
        intensity_matrix = np.log1p(intensity_matrix)
    elif intensity_transform == "minmax":
        print("⚙️ Applying min-max scaling...")
        min_val, max_val = np.min(intensity_matrix), np.max(intensity_matrix)
        intensity_matrix = (intensity_matrix - min_val) / (max_val - min_val)
    elif intensity_transform == "normalize":
        print("⚙️ Normalizing each wavelength (row)...")
        row_max = np.max(intensity_matrix, axis=1, keepdims=True)
        row_max[row_max == 0] = 1  # avoid division by zero
        intensity_matrix = intensity_matrix / row_max
    elif isinstance(intensity_transform, tuple) and intensity_transform[0] == "custom":
        func = intensity_transform[1]
        print("⚙️ Applying custom transformation...")
        intensity_matrix = func(intensity_matrix)
    else:
        print("⚙️ Using raw intensities.")

    # Create contour plot
    fig = go.Figure(data=go.Contour(
        z=intensity_matrix,
        x=rt_values,
        y=wavelengths,
        colorscale='Viridis',
        contours=dict(showlabels=True),
        colorbar=dict(title='Intensity')
    ))

    fig.update_layout(
        title="UV Matrix Contour Plot",
        xaxis_title="Retention Time (min)",
        yaxis_title="Wavelength (nm)",
        template="plotly_white"
    )

    # Save as HTML
    fig.write_html(output_html)
    print(f"✅ Contour plot saved as: {output_html}")

        
# --------------------------------------------------------------------------
#               3) DATA REFERENCING & ALIGNMENT FUNCTIONS
# --------------------------------------------------------------------------
def ref_spectra_to_df(df: pd.DataFrame,
                      thresh: float = 0.01,
                      offsetppm: float | None = None,
                      target_position: float = 0.0,
                      xlim: tuple[float, float] = (-0.7, 0.7)
                      ) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Reference (shift) all sample columns to a common RT anchor.
    First column must be 'RT(min)'. Builds a new common axis ('new_axis') by
    referencing the *first* sample to target_position, then shifts others to match.

    Parameters
    ----------
    df : DataFrame
        First column 'RT(min)', remaining columns are samples.
    thresh : float
        Peak height threshold as a fraction of the sample's max intensity (0–1).
    offsetppm : float | None
        If given, the RT (min) to use as the anchor for the first sample.
        If None, pick the tallest peak inside `xlim`.
    target_position : float
        The RT (min) where the anchor will be placed (e.g., 11.15).
    xlim : (lo, hi)
        RT window to search the anchor in the first sample.

    Returns
    -------
    referenced_df : DataFrame
        New dataframe with same columns as df; RT axis shifted so the anchor is at target_position.
    offsets : dict
        Map {sample_name: offset_rt} used for each sample.
    """
    axis_col = "RT(min)"
    if df.columns[0] != axis_col:
        raise ValueError(f"The first column must be named '{axis_col}'")

    rt_axis = df[axis_col].to_numpy()
    sample_cols = list(df.columns[1:])
    offsets: dict[str, float] = {}

    # --- 1) Determine anchor in the first sample
    s0 = sample_cols[0]
    y0 = df[s0].to_numpy()
    # search region
    lo, hi = (min(*xlim), max(*xlim))
    region_mask = (rt_axis >= lo) & (rt_axis <= hi)
    if not np.any(region_mask):
        # fallback: whole axis
        region_mask = np.ones_like(rt_axis, dtype=bool)

    # peaks above threshold in region
    th0 = float(thresh) * (np.nanmax(y0) if np.isfinite(np.nanmax(y0)) else 0.0)
    peaks, props = find_peaks(y0, height=th0)
    peaks = peaks[region_mask[peaks]]
    if peaks.size == 0:
        # fallback: use global maximum of s0 within region
        idx = int(np.argmax(y0[region_mask]))
        anchor_rt = float(rt_axis[region_mask][idx])
    else:
        if offsetppm is not None:
            # choose peak closest to provided RT
            idx = int(np.argmin(np.abs(rt_axis[peaks] - float(offsetppm))))
            anchor_rt = float(rt_axis[peaks][idx])
        else:
            # choose tallest peak in region
            idx = int(np.argmax(y0[peaks]))
            anchor_rt = float(rt_axis[peaks][idx])

    offsets[s0] = anchor_rt

    # this will be the common axis for interpolation
    new_axis = rt_axis - anchor_rt + float(target_position)

    # --- 2) Reference every sample to the first-sample anchor neighborhood
    ref_intensity: dict[str, np.ndarray] = {}
    for s in sample_cols:
        y = df[s].to_numpy()
        th = float(thresh) * (np.nanmax(y) if np.isfinite(np.nanmax(y)) else 0.0)
        pk, _ = find_peaks(y, height=th)
        if pk.size == 0:
            # if no peaks, assume zero shift
            off_rt = anchor_rt
        else:
            # pick the peak closest to the first-sample anchor_rt
            cand_rts = rt_axis[pk]
            j = int(np.argmin(np.abs(cand_rts - anchor_rt)))
            off_rt = float(cand_rts[j])

        offsets[s] = off_rt
        shifted_axis = rt_axis - off_rt + float(target_position)

        # interpolate to new_axis (keep length and alignment)
        ref_intensity[s] = np.interp(new_axis, shifted_axis, y)

    referenced_df = pd.DataFrame({axis_col: new_axis})
    for s in sample_cols:
        referenced_df[s] = ref_intensity[s]

    return referenced_df, offsets



def _fallback_align_corr_df(df: pd.DataFrame, reference: int | str = "maxcorr") -> pd.DataFrame:
    """
    Lightweight fallback alignment when pyicoshift is not available.
    Aligns each column (sample) to the reference by circular cross-correlation
    on the intensity vectors. Expects df with first col = RT(min).
    """
    if df is None or df.shape[1] < 2:
        return df

    rt = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype=float)  # shape: n_points x n_samples
    X = np.nan_to_num(X, nan=0.0)
    X = X.copy()

    # Choose reference column
    ref_idx = 0
    if isinstance(reference, int):
        ref_idx = max(0, min(X.shape[1] - 1, reference))
    elif isinstance(reference, str) and reference.lower() == "maxcorr":
        # pick the column with maximum total energy
        ref_idx = int(np.argmax(np.sum(X * X, axis=0)))

    ref = X[:, ref_idx]
    n = ref.shape[0]

    # align columns by maximizing correlation with ref
    for j in range(X.shape[1]):
        x = X[:, j]
        # compute linear correlation via FFT convolution trick
        # pad to 2n for safer wraparound
        pad = n - 1
        a = np.pad(ref, (pad, pad), mode="constant")
        b = np.pad(x, (pad, pad), mode="constant")
        c = np.correlate(a, b, mode="valid")  # length 2n-1
        shift = int(np.argmax(c) - (n - 1))
        X[:, j] = np.roll(x, shift)

    out = pd.DataFrame(X, columns=df.columns[1:])
    out.insert(0, df.columns[0], rt)
    return out



def align_samples_using_icoshift(df, n_intervals=50, target='maxcorr'):
    """
    Aligns spectral data using Icoshift if available; otherwise falls back
    to correlation-based alignment (no crash).
    - df: DataFrame with RT(min) as first column.
    - n_intervals: number of segments (icoshift only).
    - target: 'maxcorr' or index of reference sample.
    """
    # If pyicoshift is unavailable, use fallback (no ImportError)
    if not HAVE_PYICOSHIFT or ICOSHF_CALLABLE is None:
        return _fallback_align_corr_df(df, reference=target)

    ppm = df.iloc[:, 0].to_numpy()
    Xnp = df.iloc[:, 1:].to_numpy()

    if np.isnan(Xnp).all():
        # nothing useful to align
        return df

    # Icoshift expects (n_samples, n_points)
    Xnp = np.nan_to_num(Xnp.T, nan=0.0)
    try:
        result = ICOSHF_CALLABLE(Xnp, segments=n_intervals, reference='maxcorr' if target == 'maxcorr' else target)
    except TypeError:
        result = ICOSHF_CALLABLE(Xnp, segments=n_intervals)

    aligned_df = pd.DataFrame(result.T, columns=df.columns[1:])
    aligned_df.insert(0, df.columns[0], ppm)
    return aligned_df



def RAFFT_df(data, reference_idx=0, shift_RT=None, lookahead=1):
    axis = data.iloc[:, 0].values
    intensities = data.iloc[:, 1:].values
    n_points = len(axis)
    if intensities.shape[0] == n_points:
        intensities = intensities.T
    n_spectra = intensities.shape[0]
    if reference_idx < 0 or reference_idx >= n_spectra:
        raise ValueError(f"Reference index must be between 0 and {n_spectra-1}.")
    reference_spectrum = intensities[reference_idx, :]
    if shift_RT is not None:
        dppm = np.abs(axis[1] - axis[0])
        shift = int(round(shift_RT / dppm))
    else:
        shift = len(reference_spectrum)
    aligned_intensities = np.zeros_like(intensities)
    for i in range(n_spectra):
        aligned_intensities[i, :] = recur_align(intensities[i, :], reference_spectrum, shift, lookahead)
    aligned_intensities = aligned_intensities.T
    aligned_df = pd.DataFrame(np.column_stack((axis, aligned_intensities)), columns=data.columns)
    return aligned_df

# -------------------------- Helper Functions for RAFFT --------------------------
def recur_align(spectrum, reference, shift, lookahead):
    if len(spectrum) < 10:
        return spectrum
    lag = fft_corr(spectrum, reference, shift)
    if lag == 0 and lookahead <= 0:
        return spectrum
    else:
        if lag == 0:
            lookahead -= 1
        if abs(lag) < len(spectrum):
            aligned = move_seg(spectrum, lag)
        else:
            aligned = spectrum.copy()
        mid = find_mid(aligned)
        first_seg = recur_align(aligned[:mid], reference[:mid], shift, lookahead)
        second_seg = recur_align(aligned[mid:], reference[mid:], shift, lookahead)
        return np.concatenate([first_seg, second_seg])

def fft_corr(spectrum, target, shift):
    M = len(target)
    diff = 1e6
    for i in range(1, 21):
        curdiff = (2**i) - M
        if curdiff > 0 and curdiff < diff:
            diff = curdiff
    diff = int(diff)
    target_pad = np.pad(target, (0, diff), mode='constant')
    spectrum_pad = np.pad(spectrum, (0, diff), mode='constant')
    M_new = len(target_pad)
    X = np.fft.fft(target_pad)
    Y = np.fft.fft(spectrum_pad)
    R = (X * np.conjugate(Y)) / M_new
    rev = np.fft.ifft(R)
    vals = np.real(rev)
    maxi = -1
    maxpos = 0
    shift = min(shift, M_new)
    for i in range(shift):
        if vals[i] > maxi:
            maxi = vals[i]
            maxpos = i
        if vals[M_new - i - 1] > maxi:
            maxi = vals[M_new - i - 1]
            maxpos = M_new - i - 1
    if maxi < 0.1:
        return 0
    if maxpos > len(vals) / 2:
        lag = maxpos - len(vals) - 1
    else:
        lag = maxpos - 1
    return lag

def move_seg(seg, lag):
    if lag == 0 or lag >= len(seg):
        return seg
    if lag > 0:
        ins = np.full(lag, seg[0])
        return np.concatenate([ins, seg[:-lag]])
    else:
        lag_abs = abs(lag)
        ins = np.full(lag_abs, seg[-1])
        return np.concatenate([seg[lag_abs:], ins])

def find_mid(spec):
    M = int(np.ceil(len(spec) / 2))
    offset = int(np.floor(M / 4))
    start = max(M - offset, 0)
    end = min(M + offset, len(spec))
    spec_segment = spec[start:end]
    I = np.argmin(spec_segment)
    mid = I + start
    return mid

def PAFFT_df(data, segSize_RT, reference_idx=0, shift_RT=None):
    axis = data.iloc[:, 0].values
    intensities = data.iloc[:, 1:].values
    n_points = len(axis)
    if intensities.shape[0] == n_points:
        intensities = intensities.T
    n_spectra = intensities.shape[0]
    if reference_idx < 0 or reference_idx >= n_spectra:
        raise ValueError(f"Reference index must be between 0 and {n_spectra-1}.")
    reference_spectrum = intensities[reference_idx, :]
    dppm = np.abs(axis[1] - axis[0])
    if shift_RT is not None:
        shift = int(round(shift_RT / dppm))
    else:
        shift = len(reference_spectrum)
    segSize = int(round(segSize_RT / dppm))
    aligned_intensities = np.zeros_like(intensities)
    for i in range(n_spectra):
        aligned_intensities[i, :] = PAFFT(intensities[i, :], reference_spectrum, segSize, shift)
    aligned_intensities = aligned_intensities.T
    aligned_df = pd.DataFrame(np.column_stack((axis, aligned_intensities)), columns=data.columns)
    return aligned_df

def PAFFT(spectrum, reference, segSize, shift):
    n_points = len(spectrum)
    aligned_segments = []
    startpos = 0
    while startpos < n_points:
        endpos = startpos + segSize * 2
        if endpos >= n_points:
            samseg = spectrum[startpos:]
            refseg = reference[startpos:]
        else:
            samseg = spectrum[startpos + segSize: endpos - 1]
            refseg = reference[startpos + segSize: endpos - 1]
            minpos = find_min(samseg, refseg)
            endpos = startpos + minpos + segSize
            samseg = spectrum[startpos:endpos]
            refseg = reference[startpos:endpos]
        lag = fft_corr(samseg, refseg, shift)
        moved = move_seg(samseg, lag)
        aligned_segments.append(moved)
        startpos = endpos + 1
    aligned_full = np.concatenate(aligned_segments)
    if len(aligned_full) < n_points:
        aligned_full = np.pad(aligned_full, (0, n_points - len(aligned_full)), mode='edge')
    else:
        aligned_full = aligned_full[:n_points]
    return aligned_full

def find_min(samseg, refseg):
    Cs = np.sort(samseg)
    Is = np.argsort(samseg)
    Cr = np.sort(refseg)
    Ir = np.argsort(refseg)
    n_limit = max(1, int(round(len(Cs) / 20)))
    for i in range(n_limit):
        for j in range(n_limit):
            if Ir[j] == Is[i]:
                return Is[i]
    return Is[0]

def apply_alignment_by_regions(aligned_df, region_alignments):
    aligned_parts = []
    for region_info in region_alignments:
        start_ppm, end_ppm = region_info['region']
        align_func = region_info['align_func']
        params = region_info.get('params', {})
        region_mask = (aligned_df["RT(min)"] >= start_ppm) & (aligned_df["RT(min)"] <= end_ppm)
        region_df = aligned_df.loc[region_mask].copy()
        aligned_region = align_func(region_df, **params)
        aligned_parts.append(aligned_region)
    final_aligned_df = pd.concat(aligned_parts, ignore_index=True)
    final_aligned_df = final_aligned_df.sort_values("RT(min)").reset_index(drop=True)
    return final_aligned_df

# --------------------------------------------------------------------------
#               4) EXPORT FUNCTIONS
# --------------------------------------------------------------------------
def sanitize_string(s):
    return re.sub(r'[^a-zA-Z0-9_]', '', s)

def export_metaboanalyst(aligned_df, df_metadata,
                         sample_id_col="LC_filename",
                         class_col="ATTRIBUTE_classification",
                         output_file="metaboanalyst_input.csv"):
    orig_col_names = list(aligned_df.columns)
    sanitized_sample_cols = [sanitize_string(s) for s in orig_col_names[1:]]
    sanitized_col_names = [orig_col_names[0]] + sanitized_sample_cols
    aligned_df = aligned_df.copy()
    aligned_df.columns = sanitized_col_names
    sample_cols = sanitized_col_names[1:]
    meta = df_metadata.copy()
    meta[sample_id_col] = meta[sample_id_col].apply(sanitize_string)
    meta[class_col] = meta[class_col].apply(sanitize_string)
    meta_indexed = meta.set_index(sample_id_col)
    classification_series = meta_indexed.reindex(sample_cols)[class_col]
    valid_sample_cols = classification_series.dropna().index.tolist()
    if len(valid_sample_cols) < len(sample_cols):
        missing = set(sample_cols) - set(valid_sample_cols)
        print(f"Warning: The following sample IDs are missing metadata and will be excluded: {missing}")
    classification_row = [""]
    classification_row.extend(classification_series.loc[valid_sample_cols].values)
    new_df = aligned_df[['RT(min)'] + valid_sample_cols].copy()
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_df.columns.tolist())
        writer.writerow(classification_row)
        for i in range(len(new_df)):
            writer.writerow(new_df.iloc[i].values)
    print(f"MetaboAnalyst input CSV saved as: {output_file}")
    return new_df

# --------------------------------------------------------------------------
#               5) CENTERING, NORMALIZATION & SCALING FUNCTIONS
# --------------------------------------------------------------------------
def _numeric_cols(df, exclude_columns=None):
    exc = set(exclude_columns or [])
    exc.add("RT(min)")
    return [c for c in df.columns if c not in exc and pd.api.types.is_numeric_dtype(df[c])]

def standard_scale(df: pd.DataFrame, exclude_columns=None) -> pd.DataFrame:
    """
    Mean-center and unit-variance scale each numeric column (z-score).
    Numpy-only (no sklearn). Skips RT(min) and exclude_columns.
    """
    out = df.copy()
    for c in _numeric_cols(out, exclude_columns):
        x = out[c].to_numpy(dtype=float, copy=False)
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=0)
        out[c] = (x - mu) / sd if sd and np.isfinite(sd) else (x - mu)
    return out

def robust_scale(df: pd.DataFrame, exclude_columns=None) -> pd.DataFrame:
    """
    Median-center and scale by IQR (Q75-Q25). Numpy-only.
    """
    out = df.copy()
    for c in _numeric_cols(out, exclude_columns):
        x = out[c].to_numpy(dtype=float, copy=False)
        med = np.nanmedian(x)
        q25, q75 = np.nanpercentile(x, [25, 75])
        iqr = q75 - q25
        out[c] = (x - med) / iqr if iqr and np.isfinite(iqr) else (x - med)
    return out

def pareto_scale(df: pd.DataFrame, exclude_columns=None) -> pd.DataFrame:
    """
    Mean-center and divide by sqrt(std). Numpy-only.
    """
    out = df.copy()
    for c in _numeric_cols(out, exclude_columns):
        x = out[c].to_numpy(dtype=float, copy=False)
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=0)
        denom = np.sqrt(sd) if sd and np.isfinite(sd) else 1.0
        out[c] = (x - mu) / denom
    return out

def range_scale(df: pd.DataFrame, exclude_columns=None) -> pd.DataFrame:
    """
    Mean-center and divide by (max - min). Numpy-only.
    """
    out = df.copy()
    for c in _numeric_cols(out, exclude_columns):
        x = out[c].to_numpy(dtype=float, copy=False)
        mu = np.nanmean(x)
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        rng = xmax - xmin
        out[c] = (x - mu) / rng if rng and np.isfinite(rng) else (x - mu)
    return out

def min_max_scale(df: pd.DataFrame, new_min=0.0, new_max=1.0, exclude_columns=None) -> pd.DataFrame:
    """
    Rescale each numeric column to [new_min, new_max]. Numpy-only.
    """
    out = df.copy()
    a, b = float(new_min), float(new_max)
    for c in _numeric_cols(out, exclude_columns):
        x = out[c].to_numpy(dtype=float, copy=False)
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        rng = xmax - xmin
        if rng and np.isfinite(rng):
            out[c] = a + (x - xmin) * (b - a) / rng
        else:
            out[c] = x  # constant column; leave as-is
    return out

def auto_scale_m(df: pd.DataFrame, exclude_columns=None) -> pd.DataFrame:
    """
    Classic 'autoscale' (mean=0, std=1). Delegates to standard_scale (numpy-only).
    """
    return standard_scale(df, exclude_columns=exclude_columns)



# Centering Functions
def log_transform(df, constant=1):
    df = df.copy()
    for column in df.columns:
        if column != "RT(min)":
            adjusted = df[column] + constant
            adjusted = adjusted.where(adjusted > 0, np.finfo(float).eps)
            df[column] = np.log10(adjusted)
    return df

def sqrt_transform(df):
    df = df.copy()
    for column in df.columns:
        if column != "RT(min)":
            df[column] = np.sqrt(df[column].clip(lower=0))
    return df

def cbrt_transform(df):
    df = df.copy()
    for column in df.columns:
        if column != "RT(min)":
            df[column] = np.cbrt(df[column])
    return df

# Normalization Functions
def min_max_normalize(df):
    df = df.copy()
    for column in df.columns:
        if column != "RT(min)":
            min_val = df[column].min()
            max_val = df[column].max()
            df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def z_score_normalize(df, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns if exclude_columns else []
    for column in df.columns:
        if column not in exclude_columns and column != 'RT(min)':
            mean_val = df[column].mean()
            std_val = df[column].std()
            if std_val != 0:
                df[column] = (df[column] - mean_val) / std_val
            else:
                df[column] = df[column] - mean_val
    return df

def normalize_by_control(df, control_column, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns if exclude_columns else []
    control = df[control_column]
    for column in df.columns:
        if column != control_column and column not in exclude_columns and column != 'RT(min)':
            df[column] = df[column] / control
    return df

def pqn_normalize(df, reference=None, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns if exclude_columns else []
    cols = [col for col in df.columns if col != 'RT(min)' and col not in exclude_columns]
    df_numeric = df[cols]
    if reference is None:
        reference = df_numeric.median(axis=1)
    quotients = df_numeric.divide(reference, axis=0)
    median_quotients = quotients.median(axis=0)
    df_norm = df_numeric.divide(median_quotients, axis=1)
    for col in df.columns:
        if col not in df_norm.columns:
            df_norm[col] = df[col]
    df_norm = df_norm[df.columns]
    return df_norm

def std_dev_normalize(df, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns if exclude_columns else []
    for column in df.columns:
        if column not in exclude_columns and column != 'RT(min)':
            std_val = df[column].std()
            if std_val != 0:
                df[column] = df[column] / std_val
    return df

def median_normalize(df, target_median=1.0, exclude_columns=None):
    df = df.copy()
    exclude_columns = exclude_columns if exclude_columns else []
    for column in df.columns:
        if column not in exclude_columns and column != 'RT(min)':
            median_val = df[column].median()
            if median_val != 0:
                df[column] = (df[column] / median_val) * target_median
    return df

def quantile_normalize(df, exclude_columns=None):
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)", "Chemical Shift (ppm)"]
    else:
        exclude_columns = list(set(exclude_columns + ["RT(min)", "Chemical Shift (ppm)"]))
    norm_cols = [col for col in df.columns if col not in exclude_columns]
    df_numeric = df[norm_cols]
    sorted_df = pd.DataFrame(
        np.sort(df_numeric.values, axis=0),
        index=df_numeric.index,
        columns=df_numeric.columns
    )
    rank_means = sorted_df.mean(axis=1)
    df_normalized = df_numeric.copy()
    for col in df_numeric.columns:
        order = df_numeric[col].argsort()
        normalized_vals = pd.Series(rank_means.values, index=order)
        df_normalized[col] = normalized_vals.sort_index().values
    for col in exclude_columns:
        df_normalized[col] = df[col]
    df_normalized = df_normalized[df.columns]
    return df_normalized

# Scaling Functions
def pareto_scale(df, exclude_columns=None):
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)", "Chemical Shift (ppm)"]
    for column in df.columns:
        if column not in exclude_columns:
            std_val = df[column].std()
            if std_val != 0:
                df[column] = (df[column] - df[column].mean()) / np.sqrt(std_val)
            else:
                df[column] = df[column] - df[column].mean()
    return df

def range_scale(df, exclude_columns=None):
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = ["RT(min)", "Chemical Shift (ppm)"]
    for column in df.columns:
        if column not in exclude_columns:
            rng = df[column].max() - df[column].min()
            if rng != 0:
                df[column] = (df[column] - df[column].mean()) / rng
    return df


# --------------------------------------------------------------------------
#               6) DATA ANALYSIS: PCA & PLS-DA, VIP FUNCTIONS
# --------------------------------------------------------------------------
def pca_plot(normalized_df,
             df_metadata,
             sample_id_col='HPLC_filename',
             classification_col='ATTRIBUTE_classification',
             n_components=2,
             palette='hls',
             figsize=(8,6),
             show_fig=True):
    """
    Perform PCA on a normalized dataset and plot the first two principal components
    colored by group classification from the metadata.
    """
    X = normalized_df.transpose()
    # *** Center the data to mimic MATLAB's PCA centering ***
    mu = np.mean(X, axis=0)
    X = X - mu
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)
    
    pc_labels = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components,
                          columns=pc_labels,
                          index=X.index)
    
    pca_df = pca_df.merge(
        df_metadata[[sample_id_col, classification_col]],
        left_index=True,
        right_on=sample_id_col,
        how='left'
    )
    
    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=pca_df,
        x='PC1', y='PC2',
        hue=classification_col,
        palette=palette,
        s=100
    )
    plt.title("PCA Analysis Colored by Group Classification")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Group")
    
    if show_fig:
        plt.show()
    else:
        plt.close()
    
    return pca, pca_df

def plot_pca_loadings(data, pca_model, PC_choose=1, x_axis_col='RT(min)', 
                      output_dir='images', output_file=None, save_fig=True, show_fig=True):
    """
    Creates an interactive Plotly plot of PCA loadings for a chosen principal component,
    with options to save and/or display the plot.
    """
    if output_file is None:
        output_file = f'PCA_PC{PC_choose}_Loadings.html'
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_axis_col],
        y=pca_model.components_[PC_choose - 1],
        mode='lines',
        name=f'PC{PC_choose} Loading'
    ))
    fig.update_layout(
        title=f'PC{PC_choose} Loading Plot',
        xaxis_title=x_axis_col,
        yaxis_title='Loading Value',
        legend_title='Component'
    )
    
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        fig.write_html(output_path)
        print(f"Plot saved as: {output_path}")
    
    if show_fig:
        fig.show()
    
    return fig


def nipals_pca(X, n_components, thresh=1e-15):
    """
    Perform PCA using the iterative NIPALS algorithm.
    
    Parameters:
        X : np.ndarray
            Data matrix with shape (n_samples, n_features).
        n_components : int
            Number of principal components to compute.
        thresh : float, optional
            Convergence threshold for the iterative algorithm.
    
    Returns:
        T : np.ndarray
            Scores matrix with shape (n_samples, n_components).
        P : np.ndarray
            Loadings matrix with shape (n_components, n_features).
        variance : np.ndarray
            Explained variance (as fraction of total variance) for each component.
    """
    n_samples, n_features = X.shape
    T = np.zeros((n_samples, n_components))  # scores
    P = np.zeros((n_features, n_components)) # loadings (each column is a loading vector)
    variance = np.zeros(n_components)        # variance explained by each component
    
    Xi = X.copy()  # working copy of X
    total_variance = np.sum(X ** 2)
    
    for i in range(n_components):
        residual = 1.0
        p_initial = np.zeros(n_features)
        # Initialize t as the first column of Xi.
        t = Xi[:, 0].copy()
        
        while residual > thresh:
            # Compute loading: project Xi onto t.
            p = np.dot(Xi.T, t) / np.dot(t, t)
            # Normalize p to have unit length.
            norm_p = np.sqrt(np.dot(p, p))
            if norm_p != 0:
                p = p / norm_p
            # Recalculate score: project Xi onto p.
            t = np.dot(Xi, p) / np.dot(p, p)
            # Check convergence based on change in loadings.
            E = p_initial - p
            residual = np.dot(E, E)
            p_initial = p.copy()
        
        T[:, i] = t
        P[:, i] = p
        # Remove the contribution of the extracted component.
        Xi = Xi - np.outer(t, p)
        # Calculate the proportion of variance explained by this component.
        variance[i] = np.sum((np.outer(t, p))**2) / total_variance
    
    # Transpose loadings so that each row corresponds to a component.
    return T, P.T, variance

def perform_pca_analysis(data, pc_x=1, pc_y=2, n_components=None, variance_threshold=90,
                          metadata=None, color_column="ATTRIBUTE_group", sample_id_col="LC_filename",
                          output_dir='images', score_plot_filename=None, ev_plot_filename=None,
                          show_fig=True):
    """
    Performs PCA using an iterative NIPALS algorithm.
    The function computes PCA scores, loadings, explained variance, and creates interactive plots.
    
    Parameters:
        data : pd.DataFrame or np.ndarray
            Data with features as rows and samples as columns. The function transposes it so that
            each row represents a sample.
        pc_x, pc_y : int
            Principal components to plot on the x and y axes.
        n_components : int, optional
            Number of principal components to compute. If None, the number is determined based on
            the variance_threshold.
        variance_threshold : float, optional
            Minimum cumulative explained variance (in %) to determine the number of components.
        metadata : pd.DataFrame, optional
            Metadata to merge with the PCA scores DataFrame.
        color_column : str
            Column in metadata used for coloring the score plot.
        sample_id_col : str
            Column name for sample IDs in the metadata.
        output_dir : str
            Directory to save plot HTML files.
        score_plot_filename, ev_plot_filename : str, optional
            Filenames for saving the score and explained variance plots.
        show_fig : bool
            Whether to display the plots.
    
    Returns:
        tuple: (pca_model, scores_df, explained_variance) where:
            - pca_model is an object with attributes 'scores', 'loadings', 'variance', and 'components_'.
            - scores_df is a DataFrame with PCA scores (and merged metadata if provided).
            - explained_variance is an array with percentage explained variance per component.
    """
    # If data is a DataFrame, drop the "RT (ppm)" column if it exists.
    if isinstance(data, pd.DataFrame):
        if "RT (ppm)" in data.columns:
            data = data.drop(columns=["RT (ppm)"])

    # Transpose data so that rows represent samples.
    if isinstance(data, pd.DataFrame):
        X = data.transpose().values
        sample_ids = data.transpose().index
    else:
        X = data.T
        sample_ids = np.arange(X.shape[0])
    
    # *** Center the data to mimic MATLAB's PCA centering ***
    mu = np.mean(X, axis=0)
    X = X - mu

    # Determine the maximum possible number of components.
    max_comp = min(X.shape)
    
    # First, run NIPALS for the maximum number of components to calculate cumulative variance.
    T_full, P_full, var_full = nipals_pca(X, n_components=max_comp)
    cum_var = np.cumsum(var_full) * 100  # cumulative explained variance in %
    
    # Determine the number of components needed to reach the variance threshold.
    n_comp = np.argmax(cum_var >= variance_threshold) + 1
    # Ensure at least the plotted components are computed.
    n_comp = max(n_comp, pc_x, pc_y)
    
    # Override if user provided a fixed n_components.
    if n_components is not None:
        n_comp = n_components
    
    # Run NIPALS again with the desired number of components.
    scores, loadings, variance = nipals_pca(X, n_components=n_comp)
    comp_labels = [f"PC{i+1}" for i in range(n_comp)]
    
    # Create a DataFrame for PCA scores.
    scores_df = pd.DataFrame(scores, columns=comp_labels, index=sample_ids)
    
    # Merge with metadata if provided.
    if metadata is not None:
        scores_df = scores_df.merge(metadata[[sample_id_col, color_column]],
                                    left_index=True, right_on=sample_id_col, how='left')
    
    # Convert explained variance fractions to percentages.
    explained_variance = np.array(variance) * 100
    
    # Create axis labels that include the explained variance.
    xlabel = f"PC{pc_x} ({explained_variance[pc_x-1]:.1f}% explained variance)"
    ylabel = f"PC{pc_y} ({explained_variance[pc_y-1]:.1f}% explained variance)"
    
    # Create the PCA score scatter plot.
    fig_scores = px.scatter(
        scores_df, 
        x=f"PC{pc_x}", 
        y=f"PC{pc_y}",
        color=color_column if metadata is not None else None,
        title="PCA Score Plot",
        labels={
            f"PC{pc_x}": xlabel,
            f"PC{pc_y}": ylabel
        }
    )
    if score_plot_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        score_plot_file = os.path.join(output_dir, score_plot_filename)
        fig_scores.write_html(score_plot_file)
    if show_fig:
        fig_scores.show()
    
    # Create explained variance bar plot.
    ev_df = pd.DataFrame({
        "Component": comp_labels, 
        "Explained Variance (%)": explained_variance
    })
    fig_ev = px.bar(
        ev_df, 
        x="Component", 
        y="Explained Variance (%)",
        title="Explained Variance by Principal Components",
        text=ev_df["Explained Variance (%)"].apply(lambda x: f"{x:.1f}%")
    )
    fig_ev.update_traces(textposition='outside')
    if ev_plot_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        ev_plot_file = os.path.join(output_dir, ev_plot_filename)
        fig_ev.write_html(ev_plot_file)
    if show_fig:
        fig_ev.show()
    
    # Package the NIPALS results in a dictionary as the PCA model.
    model_dict = {
        "scores": scores,
        "loadings": loadings,
        "variance": variance,
        "components_": loadings
    }
    
    # Convert the dictionary to an object with attribute access.
    pca_model = SimpleNamespace(**model_dict)
    
    return pca_model, scores_df, explained_variance


def plot_pca_loadings(data, pca_model, PC_choose=1, x_axis_col='RT(min)', 
                      output_dir='images', output_file=None, save_fig=True, show_fig=True):
    """
    Creates an interactive Plotly plot of PCA loadings for a chosen principal component,
    with options to save and/or display the plot.
    """
    if output_file is None:
        output_file = f'PCA_PC{PC_choose}_Loadings.html'
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_axis_col],
        y=pca_model.components_[PC_choose - 1],
        mode='lines',
        name=f'PC{PC_choose} Loading'
    ))
    fig.update_layout(
        title=f'PC{PC_choose} Loading Plot',
        xaxis_title=x_axis_col,
        yaxis_title='Loading Value',
        legend_title='Component'
    )
    
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        fig.write_html(output_path)
        print(f"Plot saved as: {output_path}")
    
    if show_fig:
        fig.show()
    
    return fig

def plot_pca_scores(scores_df, pc_x, pc_y, explained_variance):
    fig = px.scatter(scores_df, x=f'PC{pc_x}', y=f'PC{pc_y}', text=scores_df.index, title=f'PCA Score Plot: PC{pc_x} vs PC{pc_y}')
    fig.update_layout(
        xaxis_title=f'PC{pc_x} ({explained_variance[pc_x-1]:.2f}%)',
        yaxis_title=f'PC{pc_y} ({explained_variance[pc_y-1]:.2f}%)'
    )
    fig.update_traces(marker=dict(size=7),
                      selector=dict(mode='markers+text'))
    fig.show()


def perform_pls_da(
    data,
    metadata,
    group_col="ATTRIBUTE_group",
    sample_id_col="LC_filename",
    n_components=2,
    output_dir="images",
    score_plot_filename=None,
    show_fig=True
):
    """
    Perform PLS-DA (Partial Least Squares Discriminant Analysis) by:
      1. One-hot encoding the class labels.
      2. Fitting a PLSRegression model.
      3. Plotting the first two latent variable scores, colored by group.

    Parameters
    ----------
    data : pd.DataFrame
        Normalized feature matrix with columns as samples (and rows as features).
        (If the chemical shift axis is included as a row or column, it will be ignored.)
    metadata : pd.DataFrame
        DataFrame containing sample metadata. Must include the sample_id_col (matching data columns)
        and a grouping column (group_col) for classification.
    group_col : str
        Column in metadata that contains the class/group labels (e.g., "ATTRIBUTE_group").
    sample_id_col : str
        Column in metadata that matches the sample IDs in data’s columns (e.g., "LC_filename").
    n_components : int
        Number of latent variables (components) to use in the PLS model. Default=2 for a 2D plot.
    output_dir : str
        Directory where the HTML plot will be saved if score_plot_filename is not None.
    score_plot_filename : str or None
        If provided, the interactive plot is saved as an HTML file with this name in output_dir.
    show_fig : bool
        If True, the interactive Plotly figure is displayed.

    Returns
    -------
    pls_model : PLSRegression
        The fitted PLS regression model.
    scores_df : pd.DataFrame
        DataFrame containing the latent variable scores for each sample, merged with group labels.
    """
    # If data is a DataFrame, remove any row or column named "RT (ppm)".
    if isinstance(data, pd.DataFrame):
        if "RT (ppm)" in data.columns:
            data = data.drop(columns=["Chemical Shift (ppm)"])
        if "RT (ppm)" in data.index:
            data = data.drop(index=["Chemical Shift (ppm)"])
    
    # Ensure output directory exists if we plan to save
    if score_plot_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Transpose data so rows = samples, columns = features
    X = data.transpose()  # shape = [n_samples, n_features]
    
    # 2. Align the metadata so that it matches the rows of X
    sample_index_df = pd.DataFrame({sample_id_col: X.index})
    merged_df = sample_index_df.merge(metadata, on=sample_id_col, how='left')
    
    # 3. One-hot encode the group labels for PLS-DA
    groups = merged_df[group_col].astype(str).values.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(groups)  # shape = [n_samples, n_classes]
    
    # 4. Fit the PLS regression model
    pls_model = PLSRegression(n_components=n_components)
    pls_model.fit(X, Y)
    
    # 5. Extract x-scores (latent variable scores for each sample)
    x_scores = pls_model.x_scores_
    lv_cols = [f"LV{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(x_scores, columns=lv_cols, index=X.index)
    
    # (A) Reset index and rename so sample IDs become a column named sample_id_col
    scores_df = scores_df.reset_index().rename(columns={"index": sample_id_col})
    
    # (B) Merge on the sample_id_col to get group labels
    scores_df = scores_df.merge(
        merged_df[[sample_id_col, group_col]],
        on=sample_id_col,
        how='left'
    )
    
    # 6. Plot the first two latent variables in an interactive scatter plot
    if n_components >= 2:
        fig = px.scatter(
            scores_df,
            x="LV1",
            y="LV2",
            color=group_col,
            hover_data=[sample_id_col],
            title="PLS-DA Score Plot (LV1 vs. LV2)",
            labels={"LV1": "Latent Variable 1", "LV2": "Latent Variable 2"}
        )
        if score_plot_filename is not None:
            html_path = os.path.join(output_dir, score_plot_filename)
            fig.write_html(html_path)
        if show_fig:
            fig.show()
    else:
        print(f"n_components={n_components} < 2, so no 2D score plot was generated.")
    
    return pls_model, scores_df


    
    
def plot_plsda_loadings(data, plsr_model, component=1, x_axis_col='RT(min)', 
                        output_dir='images', output_file=None, save_fig=True, show_fig=True):
    loadings = plsr_model.x_loadings_
    if component < 1 or component > loadings.shape[1]:
        raise ValueError(f"Component number must be between 1 and {loadings.shape[1]}, got {component}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_axis_col],
        y=loadings[:, component - 1],
        mode='lines',
        name=f'PLS-DA Component {component} Loading'
    ))
    fig.update_layout(
        title=f'PLS-DA Component {component} Loading Plot',
        xaxis_title=x_axis_col,
        yaxis_title='Loading Value'
    )
    if save_fig:
        if output_file is None:
            output_file = f'PLSDA_Component_{component}_Loadings.html'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        fig.write_html(output_path)
        print(f"Plot saved as: {output_path}")
    if show_fig:
        fig.show()
    return fig

def calculate_vip_scores(pls_model, X):
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    vip = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vip[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vip

def analyze_vip_scores(pls_model, X, top_n=10,
                       save_df=False, output_dir='images', output_file_df=None,
                       plot_fig=False, save_fig=False, output_file_plot=None, show_fig=True):
    vip = calculate_vip_scores(pls_model, X)
    vip_df = pd.DataFrame({'Variable': X.columns, 'VIP Score': vip})
    vip_df = vip_df.sort_values(by='VIP Score', ascending=False)
    top_vip = vip_df.head(top_n)
    if save_df:
        if output_file_df is None:
            output_file_df = 'VIP_scores.csv'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file_df)
        top_vip.to_csv(output_path, index=False)
        print(f"VIP scores saved as: {output_path}")
    if plot_fig:
        fig = px.bar(top_vip, x='Variable', y='VIP Score',
                     title='Top VIP Scores',
                     text='VIP Score')
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        if save_fig:
            if output_file_plot is None:
                output_file_plot = 'VIP_scores_plot.html'
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, output_file_plot)
            fig.write_html(plot_path)
            print(f"VIP scores plot saved as: {plot_path}")
        if show_fig:
            fig.show()
    return top_vip


def calculate_vip(pls_model):
    """
    Calculate VIP (Variable Importance in Projection) scores for a fitted PLS model.
    Assumes a single response variable.
    
    Parameters
    ----------
    pls_model : fitted PLSRegression model (from sklearn.cross_decomposition)
    
    Returns
    -------
    vip : np.array of shape (n_predictors,)
        VIP scores for each predictor.
    """
def calculate_vip(pls_model):
    t = pls_model.x_scores_          # shape: (n_samples, n_components)
    q = pls_model.y_loadings_          # shape: (n_components, n_responses)
    A = t.shape[1]                    # Actual number of components
    ssy = np.zeros(A)
    
    for a in range(A):
        ssy[a] = np.sum(t[:, a]**2) * (q[a, 0] ** 2)
    
    total_ssy = np.sum(ssy)
    
    vip = np.zeros(p)
    for j in range(p):
        sum_term = 0.0
        for a in range(A):
            norm_w_a = np.linalg.norm(w[:, a])
            sum_term += ((w[j, a] / norm_w_a) ** 2) * ssy[a]
        vip[j] = np.sqrt(p * sum_term / total_ssy)
    return vip

def plot_pls_loadings(data, pls_model, PC_choose=1, x_axis_col='RT(min)', 
                        output_dir='images', output_file=None, save_fig=True, show_fig=True):
    import os
    import plotly.graph_objects as go

    # Set a default output filename if not provided.
    if output_file is None:
        output_file = f'PLS_PC{PC_choose}_Loadings.html'
    
    # Create the Plotly figure.
    fig = go.Figure()
    
    # Convert PC_choose (1-indexed) to 0-indexed.
    comp_index = PC_choose - 1
    
    # Add a line trace using the PLS model's x_loadings_ for the selected component.
    fig.add_trace(go.Scatter(
        x=data[x_axis_col],
        y=pls_model.x_loadings_[:, comp_index],
        mode='lines',
        name=f'PLS Component {PC_choose} Loading'
    ))
    
    # Update layout with titles and labels.
    fig.update_layout(
        title=f'PLS Component {PC_choose} Loading Plot',
        xaxis_title=x_axis_col,
        yaxis_title='Loading Value',
        legend_title='Component'
    )
    
    # Save the plot if requested.
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        fig.write_html(output_path)
        print(f"Plot saved as: {output_path}")
    
    # Optionally display the plot.
    if show_fig:
        fig.show()
    
    return fig



def calculate_vip_scores(pls_model, X):
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    vip = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vip[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vip


def evaluate_plsda_components(X, y, groups=None, n_splits=5, 
                              save_fig=True, show_fig=True,
                              output_dir='images', 
                              output_file='PLSDA_Q2_R2_Scores.png',
                              figure_size=(10, 5)):
    """
    Evaluate PLS-DA performance over a range of components using cross-validation.
    
    If y is non-numeric (i.e. contains class labels), it is one-hot encoded before evaluation.
    For each number of components from 1 to the maximum allowed by both the full data and the smallest
    training set among the folds, the function computes:
      - Q² score: the mean cross-validated R² score (using GroupKFold)
      - R² score: the coefficient of determination on the full dataset.
    
    Both scores are then plotted on a single overlayed plot, with options to save and/or display the plot.
    
    Parameters:
        X (pd.DataFrame or np.array): Feature matrix.
            (If samples are in columns and features in rows, be sure to transpose beforehand if needed.)
        y (array-like or pd.Series): Target variable containing group/class labels.
            If non-numeric, y will be one-hot encoded.
        groups (array-like or None): Group labels for samples for GroupKFold.
            If None and X has an index, X.index is used.
        n_splits (int): Number of splits for GroupKFold cross-validation (default: 5).
        save_fig (bool): If True, save the combined plot as an image file.
        show_fig (bool): If True, display the plot interactively.
        output_dir (str): Directory where the plot image will be saved.
        output_file (str): Filename for the combined Q²/R² scores plot.
        figure_size (tuple): Figure size for the plot.
    
    Returns:
        q2_scores (np.array): Array of Q² scores for each number of components.
        r2_scores (np.array): Array of R² scores for each number of components.
    """
    # Use X.index as groups if groups not provided and available
    if groups is None:
        try:
            groups = X.index
        except AttributeError:
            raise ValueError("Groups not provided and X has no index attribute. Please supply groups explicitly.")
    
    # If y is a pandas Series, reindex it to match X's row order
    if isinstance(y, pd.Series):
        y = y.reindex(X.index)
    
    # If y is non-numeric, one-hot encode it
    y = np.array(y)
    if not np.issubdtype(y.dtype, np.number):
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y.reshape(-1, 1))
    
    # Create GroupKFold cross-validator
    group_kfold = GroupKFold(n_splits=n_splits)
    
    # Determine maximum number of components:
    # Overall limit: n_samples - 1 or n_features
    max_comp_full = min(X.shape[0] - 1, X.shape[1])
    # Determine the smallest training set size over folds
    fold_train_sizes = []
    for train_idx, _ in group_kfold.split(X, y, groups=groups):
        fold_train_sizes.append(len(train_idx))
    min_train_size = min(fold_train_sizes)
    max_comp_cv = min(min_train_size - 1, X.shape[1])
    
    num_components = min(max_comp_full, max_comp_cv)
    if num_components < max_comp_full:
        print(f"Limiting components to {num_components} based on the smallest training fold (size={min_train_size}).")
    
    # Initialize arrays to store scores
    q2_scores = np.zeros(num_components)
    r2_scores = np.zeros(num_components)
    
    # Loop over component numbers (from 1 to num_components)
    for i in range(1, num_components + 1):
        plsr = PLSRegression(n_components=i)
        # Q²: Cross-validated R² score
        q2 = cross_val_score(plsr, X, y, cv=group_kfold, groups=groups, scoring='r2')
        q2_scores[i - 1] = np.mean(q2)
        
        # R²: Fit on full data and predict
        plsr.fit(X, y)
        y_pred = plsr.predict(X)
        r2_scores[i - 1] = r2_score(y, y_pred)
    
    # Create combined plot for Q² and R² scores
    plt.figure(figsize=figure_size)
    comps = range(1, num_components + 1)
    plt.plot(comps, q2_scores, marker='o', label='Q² Scores')
    plt.plot(comps, r2_scores, marker='o', label='R² Scores', color='orange')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.title('PLS-DA Q² and R² Scores')
    plt.legend()
    
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined Q²/R² Scores plot saved as: {output_path}")
    if show_fig:
        plt.show()
    else:
        plt.close()
    
    return q2_scores, r2_scores


def calculate_vip(pls_model):
    """
    Calculate VIP (Variable Importance in Projection) scores for a fitted PLS model.
    Assumes a single response variable.
    
    Parameters
    ----------
    pls_model : fitted PLSRegression model (from sklearn.cross_decomposition)
    
    Returns
    -------
    vip : np.array of shape (n_predictors,)
        VIP scores for each predictor.
    """
    t = pls_model.x_scores_  # latent scores, shape (n_samples, n_components)
    w = pls_model.x_weights_ # weights, shape (n_predictors, n_components)
    q = pls_model.y_loadings_  # y loadings, shape (n_components, 1) for single response
    A = w.shape[1]
    p = w.shape[0]
    
    # Calculate the explained sum of squares for each component.
    ssy = np.zeros(A)
    for a in range(A):
        # For a single response, q[a,0] is used.
        ssy[a] = np.sum(t[:, a]**2) * (q[a, 0] ** 2)
    total_ssy = np.sum(ssy)
    
    vip = np.zeros(p)
    for j in range(p):
        sum_term = 0.0
        for a in range(A):
            norm_w_a = np.linalg.norm(w[:, a])
            sum_term += ((w[j, a] / norm_w_a) ** 2) * ssy[a]
        vip[j] = np.sqrt(p * sum_term / total_ssy)
    return vip


def plot_pca_loadings_with_spectra(data, normalized_df, pca_model, PC_choose=1, 
                                   x_axis_col='RT (min)', gap=100,
                                   spectra_scale=0.1,
                                   output_dir='images', output_file=None, 
                                   save_fig=True, show_fig=True):
    """
    Creates a Plotly figure that overlays the PCA loading (for a specified principal component)
    with the original normalized spectra in the background. The spectra are vertically offset 
    by a specified gap (default 100 units) and scaled by spectra_scale so that their intensity 
    better matches the PCA loading plot.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the x-axis values (e.g. rt_min).
                             Used for the PCA loading trace.
        normalized_df (pd.DataFrame): DataFrame with the original normalized spectra. It must include
                                      the same x-axis column (x_axis_col) and one or more spectra columns.
        pca_model: Fitted PCA model from which the loadings will be extracted.
        PC_choose (int): Principal component number for which to plot the loading.
        x_axis_col (str): Name of the x-axis column (default 'RT (min)').
        gap (float): Vertical gap (offset) between successive spectra.
        spectra_scale (float): Factor to multiply the normalized spectra intensities by.
        output_dir (str): Directory to save the plot HTML file.
        output_file (str or None): Filename for the output HTML file. If None, a default name is generated.
        save_fig (bool): If True, save the figure as an HTML file.
        show_fig (bool): If True, display the interactive figure.
    
    Returns:
        fig: The Plotly figure object.
    """
    if output_file is None:
        output_file = f'PCA_PC{PC_choose}_Loadings_with_Spectra.html'
        
    fig = go.Figure()

    # Add the original normalized spectra traces (background)
    # We assume normalized_df has the x-axis column, and all other columns are spectra.
    spectra_cols = [col for col in normalized_df.columns if col != x_axis_col]
    for i, col in enumerate(spectra_cols):
        offset = i * gap
        fig.add_trace(go.Scatter(
            x=normalized_df[x_axis_col],
            y=(normalized_df[col] * spectra_scale) + offset,
            mode='lines',
            line=dict(color='gray'),
            name=f'{col} (offset {offset})',
            opacity=0.6,
            showlegend=False  # Hide individual legend entries for clarity
        ))

    # Add the PCA loading trace (overlayed on top)
    fig.add_trace(go.Scatter(
        x=data[x_axis_col],
        y=pca_model.components_[PC_choose - 1],
        mode='lines',
        line=dict(color='blue', width=3),
        name=f'PC{PC_choose} Loading'
    ))
    
    fig.update_layout(
        title=f'PC{PC_choose} Loading Plot with Original Spectra Overlay',
        xaxis_title=x_axis_col,
        yaxis_title='Loading Value / Offset Spectra',
        legend_title='Component',
        xaxis=dict(autorange='reversed')
    )
    
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        fig.write_html(output_path)
        print(f"Plot saved as: {output_path}")
    
    if show_fig:
        fig.show()
    
    return fig


def perform_hca(
    data,
    metadata,
    group_col="ATTRIBUTE_group",
    sample_id_col="ATTRIBUTE_localsampleid",
    method="ward",
    metric="euclidean",
    n_clusters=None,
    output_dir="images",
    dendrogram_filename=None,
    show_fig=True
):
    """
    Perform Hierarchical Cluster Analysis (HCA) on the sample data and generate an interactive dendrogram.
    
    Parameters
    ----------
    data : pd.DataFrame
        Normalized feature matrix with columns as samples and rows as features.
    metadata : pd.DataFrame
        Metadata DataFrame containing sample information. Must include the sample_id_col and group_col.
    group_col : str
        Column in metadata that contains group labels.
    sample_id_col : str
        Column in metadata that matches the sample IDs in data’s columns.
    method : str
        The linkage algorithm to use (e.g., "ward", "single", "complete"). Default is "ward".
    metric : str
        The distance metric to use (e.g., "euclidean", "cosine"). Default is "euclidean".
    n_clusters : int or None
        If provided, the function will assign cluster memberships using the specified number of clusters.
    output_dir : str
        Directory where the dendrogram HTML file will be saved if dendrogram_filename is provided.
    dendrogram_filename : str or None
        If provided, the interactive dendrogram is saved as an HTML file with this name in output_dir.
    show_fig : bool
        If True, the interactive Plotly dendrogram is displayed.
    
    Returns
    -------
    result_dict : dict
        Dictionary containing:
          - "dendrogram_fig": The Plotly figure for the dendrogram.
          - "linkage_matrix": The linkage matrix computed for clustering.
          - "cluster_assignments": (Optional) A pandas Series mapping sample IDs to their cluster labels (if n_clusters is provided).
    """
    import os
    import pandas as pd
    from scipy.cluster.hierarchy import linkage, fcluster
    import plotly.figure_factory as ff

    # Transpose data so that rows represent samples
    X = data.drop(columns="RT (min)").transpose()  # shape: [n_samples, n_features]
    
    # Compute the linkage matrix using the specified method and metric
    Z = linkage(X, method=method, metric=metric)
    
    # Create a dendrogram figure using Plotly's figure factory
    # The labels are the sample IDs from X.index.
    dendro_fig = ff.create_dendrogram(
        X.values,
        orientation='left',
        labels=list(X.index),
        linkagefun=lambda x: Z,
        color_threshold=0  # Let Plotly choose the default color cutoff
    )
    
    # Merge metadata for hover labels
    meta_lookup = metadata.set_index(sample_id_col)[group_col].to_dict()
    for trace in dendro_fig.data:
        if trace.text is not None:
            trace.text = [f"{lbl}<br>Group: {meta_lookup.get(lbl, 'NA')}" for lbl in trace.text]
    
    # If n_clusters is specified, assign clusters using fcluster.
    cluster_assignments = None
    if n_clusters is not None:
        clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
        # Map sample IDs to clusters
        cluster_assignments = pd.Series(clusters, index=X.index, name='Cluster')
        print("Cluster assignments:")
        print(cluster_assignments)
    
    # Save the dendrogram if a filename is provided.
    if dendrogram_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, dendrogram_filename)
        dendro_fig.write_html(output_path)
        print(f"Dendrogram saved as: {output_path}")
    
    if show_fig:
        dendro_fig.show()
    
    result_dict = {
        "dendrogram_fig": dendro_fig,
        "linkage_matrix": Z,
        "cluster_assignments": cluster_assignments
    }
    return result_dict

def perform_opls_da(
    data,
    metadata,
    group_col="ATTRIBUTE_group",
    sample_id_col="LC_filename",
    n_components=2,
    output_dir="images",
    score_plot_filename=None,
    show_fig=True
):
    """
    Perform Orthogonal PLS-DA (OPLS-DA) by:
      1. One-hot encoding the class labels.
      2. Fitting a PLSRegression model with one predictive component.
      3. Removing the predictive variation to get orthogonal components using PCA on the residual matrix.
      4. Plotting the predictive (LV1) vs. the first orthogonal component (LV2).

    Parameters
    ----------
    data : pd.DataFrame
        Normalized feature matrix with columns as samples (rows as features).
        (If the rt_min axis is included as a column or row, it will be ignored.)
    metadata : pd.DataFrame
        DataFrame containing sample metadata. Must include the sample_id_col (matching data columns)
        and a grouping column (group_col) for classification.
    group_col : str
        Column in metadata that contains the class/group labels.
    sample_id_col : str
        Column in metadata that matches the sample IDs in data’s columns.
    n_components : int
        Total number of components: 1 predictive + (n_components - 1) orthogonal.
        Default is 2 so that a 2D score plot can be generated.
    output_dir : str
        Directory where the HTML plot will be saved if score_plot_filename is provided.
    score_plot_filename : str or None
        If provided, the interactive score plot is saved as an HTML file with this name in output_dir.
    show_fig : bool
        If True, the interactive Plotly figure is displayed.

    Returns
    -------
    model_dict : dict
        Dictionary containing:
          - "pls_model": the fitted PLSRegression model (predictive component).
          - "pca_ortho": the PCA model fitted on the residual (orthogonal components) (or None if n_components==1).
          - "scores_df": DataFrame of latent variable scores (predictive and orthogonal) merged with metadata.
    """
    # If data is a DataFrame, remove any column or row named "RT (ppm)"
    if isinstance(data, pd.DataFrame):
        if "RT (ppm)" in data.columns:
            data = data.drop(columns=["RT (min)"])
        if "RT (ppm)" in data.index:
            data = data.drop(index=["RT (min)"])
    
    # Ensure output directory exists if saving the plot.
    if score_plot_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Transpose data so that rows are samples and columns are features.
    X = data.transpose()  # shape: [n_samples, n_features]

    # Align metadata with the sample IDs from X.
    sample_index_df = pd.DataFrame({sample_id_col: X.index})
    merged_df = sample_index_df.merge(metadata, on=sample_id_col, how='left')

    # One-hot encode the group labels.
    groups = merged_df[group_col].astype(str).values.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(groups)  # shape: [n_samples, n_classes]

    # --- Step 1: Fit PLSRegression for the predictive component ---
    pls_model = PLSRegression(n_components=1)
    pls_model.fit(X, Y)
    t_pred = pls_model.x_scores_      # predictive scores; shape: [n_samples, 1]
    p_pred = pls_model.x_loadings_    # predictive loadings; shape: [n_features, 1]

    # --- Step 2: Remove predictive variation to compute orthogonal variation ---
    # Reconstruct X from the predictive component.
    X_pred = t_pred.dot(p_pred.T)
    # Residual matrix (X_res) contains the orthogonal variation.
    X_res = X - X_pred

    # --- Step 3: Extract orthogonal component(s) from the residual using PCA ---
    n_ortho = n_components - 1  # number of orthogonal components
    if n_ortho > 0:
        pca_ortho = PCA(n_components=n_ortho)
        t_ortho = pca_ortho.fit_transform(X_res)  # shape: [n_samples, n_ortho]
    else:
        pca_ortho = None
        t_ortho = np.empty((X.shape[0], 0))
    
    # --- Step 4: Build the scores DataFrame ---
    # Concatenate the predictive score with orthogonal scores.
    scores = np.concatenate([t_pred, t_ortho], axis=1)  # shape: [n_samples, n_components]
    lv_names = ['Predictive'] + [f"Orthogonal{i+1}" for i in range(n_ortho)]
    scores_df = pd.DataFrame(scores, columns=lv_names, index=X.index)
    scores_df = scores_df.reset_index().rename(columns={"index": sample_id_col})
    scores_df = scores_df.merge(
        merged_df[[sample_id_col, group_col]],
        on=sample_id_col,
        how='left'
    )
    
    # --- Step 5: Plot the score plot (Predictive vs. first Orthogonal) ---
    if n_components >= 2:
        fig = px.scatter(
            scores_df,
            x="Predictive",
            y="Orthogonal1",
            color=group_col,
            hover_data=[sample_id_col],
            title="OPLS-DA Score Plot (Predictive vs. Orthogonal)",
            labels={"Predictive": "Predictive Component", "Orthogonal1": "Orthogonal Component 1"}
        )
        if score_plot_filename is not None:
            html_path = os.path.join(output_dir, score_plot_filename)
            fig.write_html(html_path)
        if show_fig:
            fig.show()
    else:
        print("n_components < 2, so no 2D score plot was generated.")
    
    model_dict = {"pls_model": pls_model, "pca_ortho": pca_ortho, "scores_df": scores_df}
    return model_dict


def plot_oplsda_predictive_loadings(data, model_dict, x_axis_col='RT(min)', 
                                    output_dir='images', output_file=None, 
                                    save_fig=True, show_fig=True):
    """
    Plot the predictive loadings from the OPLS-DA model.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the x-axis values (e.g., retention time, chemical shift).
    model_dict : dict
        Dictionary returned by perform_opls_da.
    x_axis_col : str
        Column in data to use for the x-axis.
    output_dir : str
        Directory where the plot will be saved.
    output_file : str or None
        Filename for saving the plot. Defaults to 'OPLSDA_Predictive_Loadings.html' if None.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The predictive loadings plot.
    """
    # Predictive loadings from the PLS model.
    loadings = model_dict["pls_model"].x_loadings_
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_axis_col],
        y=loadings[:, 0],
        mode='lines',
        name='OPLS-DA Predictive Loading'
    ))
    fig.update_layout(
        title='OPLS-DA Predictive Loading Plot',
        xaxis_title=x_axis_col,
        yaxis_title='Loading Value',
        xaxis=dict(autorange='reversed')
    )
    if save_fig:
        if output_file is None:
            output_file = 'OPLSDA_Predictive_Loadings.html'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        fig.write_html(output_path)
        print(f"Predictive loadings plot saved as: {output_path}")
    if show_fig:
        fig.show()
    return fig

def plot_oplsda_orthogonal_loadings(data, model_dict, component=1, x_axis_col='RT(min)', 
                                    output_dir='images', output_file=None, 
                                    save_fig=True, show_fig=True):
    """
    Plot the orthogonal loadings from the OPLS-DA model (from the PCA on the residual).
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the x-axis values (e.g., retention time, chemical shift).
    model_dict : dict
        Dictionary returned by perform_opls_da.
    component : int
        The orthogonal component number to plot (1-indexed).
    x_axis_col : str
        Column in data to use for the x-axis.
    output_dir : str
        Directory where the plot will be saved.
    output_file : str or None
        Filename for saving the plot. Defaults to 'OPLSDA_Orthogonal_Component_1_Loadings.html' if None.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The orthogonal loadings plot.
    """
    pca_ortho = model_dict.get("pca_ortho", None)
    if pca_ortho is None:
        raise ValueError("No orthogonal components are available in the OPLS-DA model.")
    
    if component < 1 or component > pca_ortho.components_.shape[0]:
        raise ValueError(f"Component must be between 1 and {pca_ortho.components_.shape[0]}, got {component}")
    
    # PCA components are stored in rows; select the desired orthogonal component.
    loadings = pca_ortho.components_[component - 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_axis_col],
        y=loadings,
        mode='lines',
        name=f'OPLS-DA Orthogonal Component {component} Loading'
    ))
    fig.update_layout(
        title=f'OPLS-DA Orthogonal Component {component} Loading Plot',
        xaxis_title=x_axis_col,
        yaxis_title='Loading Value',
        xaxis=dict(autorange='reversed')
    )
    if save_fig:
        if output_file is None:
            output_file = f'OPLSDA_Orthogonal_Component_{component}_Loadings.html'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        fig.write_html(output_path)
        print(f"Orthogonal loadings plot saved as: {output_path}")
    if show_fig:
        fig.show()
    return fig

def calculate_opls_vip_scores(model_dict, X):
    """
    Calculate VIP (Variable Importance in Projection) scores for the predictive component
    from the OPLS-DA model.
    
    Parameters
    ----------
    model_dict : dict
        Dictionary returned by perform_opls_da.
    X : pd.DataFrame
        DataFrame of features (with samples as rows). The columns should correspond to the variables.
    
    Returns
    -------
    vip : np.array
        Array of VIP scores for each variable.
    """
    pls_model = model_dict["pls_model"]
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    vip = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vip[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vip

def analyze_opls_vip_scores(model_dict, X, top_n=10,
                            save_df=False, output_dir='images', output_file_df=None,
                            plot_fig=False, save_fig=False, output_file_plot=None, show_fig=True):
    """
    Analyze and optionally save/plot the top VIP scores for the predictive component from the OPLS-DA model.
    
    Parameters
    ----------
    model_dict : dict
        Dictionary returned by perform_opls_da.
    X : pd.DataFrame
        Feature DataFrame with variables as columns.
    top_n : int
        Number of top VIP scores to return.
    save_df : bool
        Whether to save the VIP scores as a CSV file.
    output_dir : str
        Directory for saving outputs.
    output_file_df : str or None
        Filename for saving the VIP scores CSV.
    plot_fig : bool
        Whether to plot the VIP scores.
    save_fig : bool
        Whether to save the VIP scores plot.
    output_file_plot : str or None
        Filename for the VIP scores plot.
    show_fig : bool
        Whether to display the plot.
    
    Returns
    -------
    top_vip : pd.DataFrame
        DataFrame of the top VIP scores.
    """
    vip = calculate_opls_vip_scores(model_dict, X)
    vip_df = pd.DataFrame({'Variable': X.columns, 'VIP Score': vip})
    vip_df = vip_df.sort_values(by='VIP Score', ascending=False)
    top_vip = vip_df.head(top_n)
    if save_df:
        if output_file_df is None:
            output_file_df = 'OPLS_VIP_scores.csv'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file_df)
        top_vip.to_csv(output_path, index=False)
        print(f"VIP scores saved as: {output_path}")
    if plot_fig:
        fig = px.bar(top_vip, x='Variable', y='VIP Score',
                     title='Top VIP Scores (OPLS-DA)',
                     text='VIP Score')
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        if save_fig:
            if output_file_plot is None:
                output_file_plot = 'OPLS_VIP_scores_plot.html'
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, output_file_plot)
            fig.write_html(plot_path)
            print(f"VIP scores plot saved as: {plot_path}")
        if show_fig:
            fig.show()
    return top_vip


def plot_hca_heatmap(
    data,
    metadata,
    sample_id_col="LC_filename",
    group_col="ATTRIBUTE_group",
    output_dir="images",
    heatmap_filename=None,
    show_fig=True
):
    """
    Generate a heatmap of the feature matrix with samples clustered by hierarchical clustering.
    Samples are ordered based on the dendrogram.
    
    Parameters
    ----------
    data : pd.DataFrame
        Normalized feature matrix with columns as samples.
    metadata : pd.DataFrame
        Metadata DataFrame containing sample information.
    sample_id_col : str
        Column in metadata that contains sample IDs.
    group_col : str
        Column in metadata that contains group labels.
    output_dir : str
        Directory where the heatmap HTML file will be saved if heatmap_filename is provided.
    heatmap_filename : str or None
        If provided, the heatmap is saved as an HTML file with this name in output_dir.
    show_fig : bool
        If True, the interactive heatmap is displayed.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The heatmap figure.
    """
    import plotly.express as px
    # Transpose data so that rows are samples
    X = data.transpose()  # shape: [n_samples, n_features]
    
    # Compute hierarchical clustering order
    Z = linkage(X, method="ward", metric="euclidean")
    # Use the dendrogram to get the order of sample indices
    dendro = ff.create_dendrogram(X.values, orientation='left', linkagefun=lambda x: Z)
    ordered_sample_ids = dendro['layout']['yaxis']['ticktext']
    
    # Reorder the DataFrame based on the dendrogram order
    X_ordered = X.loc[ordered_sample_ids]
    
    # Add group information from metadata for annotation
    meta_lookup = metadata.set_index(sample_id_col)[group_col].to_dict()
    group_labels = [meta_lookup.get(sid, 'NA') for sid in X_ordered.index]
    
    # Create a heatmap using Plotly Express
    fig = px.imshow(
        X_ordered,
        labels=dict(x="Features", y="Samples", color="Intensity"),
        x=X_ordered.columns,
        y=X_ordered.index,
        aspect="auto",
        title="HCA Heatmap"
    )
    # Add group labels as hover information
    fig.update_traces(
        hovertemplate="<b>Sample</b>: %{y}<br><b>Group</b>: " +
                      "%{customdata}<br><b>Feature</b>: %{x}<br><b>Intensity</b>: %{z}"
    )
    fig.update_traces(customdata=np.array(group_labels).reshape(-1, 1))
    
    if heatmap_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, heatmap_filename)
        fig.write_html(output_path)
        print(f"Heatmap saved as: {output_path}")
    
    if show_fig:
        fig.show()
    
    return fig

# --------------------------------------------------------------------------
#               9) STOCSY FUNCTIONS
# --------------------------------------------------------------------------
def STOCSY_LC(target, X, rt_values):
    import mpld3
    import math
    from matplotlib.collections import LineCollection
    from scipy import stats
    
    # Determine the target vector
    if isinstance(target, float):
        idx = np.abs(rt_values - target).idxmin()
        target_vect = X.iloc[idx]
    else:
        target_vect = target

    # Compute correlation and covariance using z-scored vectors.
    corr = (stats.zscore(target_vect.T, ddof=1) @ stats.zscore(X.T, ddof=1)) / (X.T.shape[0] - 1)
    covar = (target_vect - target_vect.mean()) @ (X.T - np.tile(X.T.mean(), (X.T.shape[0], 1))) / (X.T.shape[0] - 1)
    
    # Create line segments for a continuous trace
    x = np.linspace(0, len(covar), len(covar))
    y = covar
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    fig, axs = plt.subplots(1, 1, figsize=(16, 4), sharex=True, sharey=True)
    norm = plt.Normalize(corr.min(), corr.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(corr)
    lc.set_linewidth(2)
    axs.add_collection(lc)
    fig.colorbar(lc, ax=axs)
    
    axs.set_xlim(x.min(), x.max())
    axs.set_ylim(y.min(), y.max())
    #axs.invert_xaxis()
    
    # Determine the min and max of the retention times
    min_rt = rt_values.min()
    max_rt = rt_values.max()
    ticksx = []
    tickslabels = []
    if max_rt < 30:
        ticks = np.linspace(math.ceil(min_rt), int(max_rt), int(max_rt) - math.ceil(min_rt) + 1)
    else:
        ticks = np.linspace(math.ceil(min_rt / 10.0) * 10,
                             math.ceil(max_rt / 10.0) * 10 - 10,
                             math.ceil(max_rt / 10.0) - math.ceil(min_rt / 10.0))
    currenttick = 0
    # Loop over each retention time value.
    for rt_val in rt_values:
        if currenttick < len(ticks) and rt_val > ticks[currenttick]:
            position = int((rt_val - min_rt) / (max_rt - min_rt) * x.max())
            if position < len(x):
                ticksx.append(x[position])
                tickslabels.append(ticks[currenttick])
            currenttick += 1
    plt.xticks(ticksx, tickslabels, fontsize=12)
    
    axs.set_xlabel('Retention time (ppm)', fontsize=14)
    axs.set_ylabel(f"Covariance with \n signal at {target:.2f} min", fontsize=14)
    axs.set_title(f'STOCSY from signal at {target:.2f} min', fontsize=16)
    
    text = axs.text(1, 1, '')
    lnx = plt.plot([60, 60], [0, 1.5], color='black', linewidth=0.3)
    lny = plt.plot([0, 100], [1.5, 1.5], color='black', linewidth=0.3)
    lnx[0].set_linestyle('None')
    lny[0].set_linestyle('None')
    
    def hover(event):
        if event.inaxes == axs:
            inv = axs.transData.inverted()
            maxcoord = axs.transData.transform((x[0], 0))[0]
            mincoord = axs.transData.transform((x[-1], 0))[0]
            rt_val = ((maxcoord - mincoord) - (event.x - mincoord)) / (maxcoord - mincoord) * (max_rt - min_rt) + min_rt
            cov_val = covar[int(((maxcoord - mincoord) - (event.x - mincoord)) / (maxcoord - mincoord) * len(covar))]
            cor_val = corr[int(((maxcoord - mincoord) - (event.x - mincoord)) / (maxcoord - mincoord) * len(corr))]
            text.set_visible(True)
            text.set_position((event.xdata, event.ydata))
            text.set_text(f'{rt_val:.2f} min, covariance: {cov_val:.6f}, correlation: {cor_val:.2f}')
            lnx[0].set_data([event.xdata, event.xdata], [-1, 1])
            lnx[0].set_linestyle('--')
            lny[0].set_data([x[0], x[-1]], [cov_val, cov_val])
            lny[0].set_linestyle('--')
        else:
            text.set_visible(False)
            lnx[0].set_linestyle('None')
            lny[0].set_linestyle('None')
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()
    
    if not os.path.exists('images'):
        os.mkdir('images')
    plt.savefig(f"images/stocsy_from_{target}min.pdf", transparent=True, dpi=300)
    
    html_str = mpld3.fig_to_html(fig)
    with open(f"images/stocsy_interactive_{target}min.html", "w") as f:
        f.write(html_str)
    
    return corr, covar


def STOCSY_interactive(target, X, min):
    if isinstance(target, float):
        idx = np.argmin(np.abs(np.array(min) - target))
        target_vect = X.iloc[idx]
    else:
        target_vect = target
    corr = (stats.zscore(target_vect.T, ddof=1) @ stats.zscore(X.T, ddof=1)) / ((X.T.shape[0]) - 1)
    covar = (target_vect - target_vect.mean()) @ (X.T - np.tile(X.T.mean(), (X.T.shape[0], 1))) / ((X.T.shape[0]) - 1)
    df = pd.DataFrame({'min': min, 'covar': covar, 'corr': corr})
    df = df.sort_values('min', ascending=False)
    fig = go.Figure(data=go.Scatter(
        x=df['min'],
        y=df['covar'],
        mode='lines+markers',
        marker=dict(
            color=df['corr'],
            colorscale='jet',
            colorbar=dict(title='Correlation'),
            size=6
        ),
        line=dict(width=2),
        hovertemplate='RT: %{x:.2f} min<br>Covariance: %{y:.6f}<br>Correlation: %{marker.color:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title=f'STOCSY from signal at {target:.2f} min',
        xaxis_title='min',
        yaxis_title=f"Covariance with signal at {target:.2f} min",
        xaxis_autorange='reversed'
    )
    fig.show()
    return corr, covar

def STOCSY_LC_mode(target, X, rt_values, mode="linear"):
    """
    Structured STOCSY: Compute correlation (or similarity), R², and RMSE between a target signal and a matrix of signals.
    
    Parameters:
    ----------
    target : float or Series
        Target retention time (if float) or signal vector for STOCSY anchor.
    X : DataFrame
        Data matrix where each row is a signal (e.g., from LC-MS).
    rt_values : Series
        Retention time values corresponding to each row in X.
    mode : str, optional
        Type of structured correlation model to use:
            - 'linear'       : Pearson correlation (default).
            - 'exponential'  : Exponential decay (kinetics, degradation).
            - 'sinusoidal'   : Sine wave (circadian rhythms, periodicity; uses FFT to estimate frequency).
            - 'sigmoid'      : Logistic dose-response (biological saturation, enzyme kinetics; captures thresholds and plateaus).
            - 'gaussian'     : Symmetric bell curve (chromatographic peaks, transient events).
            - 'fft'          : Frequency domain similarity (hidden periodic structure, using cosine similarity of FFT magnitudes).
            - 'polynomial'   : 2nd-degree polynomial trend (curvilinear responses).
            - 'piecewise'    : Piecewise linear (threshold, on/off behavior).
            - 'skewed_gauss' : Asymmetric bell-shaped peak (real-world chromatographic peaks).
    
    Returns:
    -------
    corr : array
        Array of correlation (or similarity) values between the target and each signal.
    covar : array
        Array of covariance values between the target and each signal.
    """
    import os
    import mpld3
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from scipy import stats
    from scipy.optimize import curve_fit
    from scipy.spatial.distance import cosine
    from scipy.fft import fft
    from scipy.stats import skewnorm

    # Define models
    def exp_model(x, a, b, c):
        return a * np.exp(-b * x) + c

    def sin_model(x, a, b, c, d):
        return a * np.sin(b * x + c) + d

    def sigmoid_model(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def gauss_model(x, a, mu, sigma, c):
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

    def poly_model(x, a0, a1, a2):
        return a0 + a1 * x + a2 * x**2

    def piecewise_model(x, x0, k1, b1, k2, b2):
        return np.piecewise(x, [x < x0, x >= x0],
                            [lambda x: k1 * x + b1, lambda x: k2 * x + b2])
    
    def skewed_gauss_model(x, a, loc, scale, alpha, offset):
        # skewnorm.pdf returns density; multiplicative factor a and offset applied.
        return a * skewnorm.pdf(x, alpha, loc, scale) + offset

    def fft_magnitude(signal):
        return np.abs(fft(signal))

    # Determine target vector
    if isinstance(target, float):
        idx = np.abs(rt_values - target).idxmin()
        target_vect = X.iloc[idx]
    else:
        target_vect = target

    corr_list = []
    r2_list = []
    rmse_list = []
    target_vals = target_vect.values

    # Loop por cada sinal em X
    for i in range(X.shape[0]):
        y = X.iloc[i].values
        x = target_vals  # usamos o mesmo vetor de x do sinal target

        try:
            if mode == "linear":
                r = np.corrcoef(x, y)[0, 1]
                fitted = y

            elif mode == "exponential":
                popt, _ = curve_fit(exp_model, x, y, maxfev=10000)
                fitted = exp_model(x, *popt)
                r = np.corrcoef(y, fitted)[0, 1]

            elif mode == "sinusoidal":
                if np.std(y) < 1e-6:
                    raise ValueError("Low signal variance")
                # Remover tendência e estimar frequência dominante via FFT
                y_detrended = y - np.mean(y)
                fft_y = fft(y_detrended)
                freqs = np.fft.fftfreq(len(x), d=(x[1] - x[0]))
                # Ignorar a frequência 0
                dom_freq_index = np.argmax(np.abs(fft_y[1:])) + 1
                dom_freq = np.abs(freqs[dom_freq_index])
                guess_freq = 2 * np.pi * dom_freq if dom_freq > 0 else 1 / (2 * np.pi)
                popt, _ = curve_fit(sin_model, x, y, p0=[1, guess_freq, 0, 0], maxfev=10000)
                fitted = sin_model(x, *popt)
                r = np.corrcoef(y, fitted)[0, 1]

            elif mode == "sigmoid":
                if np.max(x) - np.min(x) == 0 or np.max(y) - np.min(y) == 0:
                    raise ValueError("Flat signal")
                x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
                y_scaled = (y - np.min(y)) / (np.max(y) - np.min(y))
                popt, _ = curve_fit(sigmoid_model, x_scaled, y_scaled, p0=[1, 1, 0.5], maxfev=10000)
                fitted_scaled = sigmoid_model(x_scaled, *popt)
                fitted = fitted_scaled * (np.max(y) - np.min(y)) + np.min(y)
                r = np.corrcoef(y, fitted)[0, 1]

            elif mode == "gaussian":
                mu_init = x[np.argmax(y)]
                half_max = np.max(y) / 2
                indices = np.where(y > half_max)[0]
                if len(indices) > 1:
                    sigma_init = (x[indices[-1]] - x[indices[0]]) / 2.355  # FWHM ~2.355σ
                else:
                    sigma_init = np.std(x)
                popt, _ = curve_fit(gauss_model, x, y, p0=[1, mu_init, sigma_init, 0], maxfev=10000)
                fitted = gauss_model(x, *popt)
                r = np.corrcoef(y, fitted)[0, 1]

            elif mode == "fft":
                # Compute FFT magnitudes and compare via cosine similarity
                target_fft = fft_magnitude(target_vals)
                y_fft = fft_magnitude(y)
                similarity = 1 - cosine(target_fft, y_fft)
                r = similarity
                fitted = np.zeros_like(y)  # Não há ajuste, apenas similaridade

            elif mode == "polynomial":
                popt, _ = curve_fit(poly_model, x, y, p0=[0, 1, 0], maxfev=10000)
                fitted = poly_model(x, *popt)
                r = np.corrcoef(y, fitted)[0, 1]

            elif mode == "piecewise":
                # Estimativa inicial: ponto de mudança no meio do domínio
                x0_guess = np.median(x)
                popt, _ = curve_fit(piecewise_model, x, y, p0=[x0_guess, 1, 0, 0.5, 0], maxfev=10000)
                fitted = piecewise_model(x, *popt)
                r = np.corrcoef(y, fitted)[0, 1]

            elif mode == "skewed_gauss":
                loc_init = x[np.argmax(y)]
                scale_init = np.std(x)
                popt, _ = curve_fit(skewed_gauss_model, x, y, p0=[1, loc_init, scale_init, 4, 0], maxfev=10000)
                fitted = skewed_gauss_model(x, *popt)
                r = np.corrcoef(y, fitted)[0, 1]

            else:
                raise ValueError("Invalid mode")
        except Exception:
            r = 0
            fitted = np.zeros_like(y)

        r2 = r ** 2
        rmse = np.sqrt(np.mean((y - fitted) ** 2)) if mode != "fft" else np.nan
        corr_list.append(r)
        r2_list.append(r2)
        rmse_list.append(rmse)

    corr = np.array(corr_list)
    covar = (target_vect - target_vect.mean()) @ (X.T - np.tile(X.T.mean(), (X.T.shape[0], 1))) / (X.T.shape[0] - 1)

    # Plotting dos resultados (mantido igual, com eixo X invertido)
    x_plot = np.linspace(0, len(covar), len(covar))
    y_plot = covar
    points = np.array([x_plot, y_plot]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(16, 4), sharex=True, sharey=True)
    norm = plt.Normalize(corr.min(), corr.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(corr)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax)

    ax.set_xlim(x_plot.min(), x_plot.max())
    ax.set_ylim(y_plot.min(), y_plot.max())
    #ax.invert_xaxis()

    # Configurar os ticks do eixo X com base nos rt_values
    min_rt = rt_values.min()
    max_rt = rt_values.max()
    ticksx = []
    tickslabels = []
    if max_rt < 30:
        ticks = np.linspace(math.ceil(min_rt), int(max_rt), int(max_rt) - math.ceil(min_rt) + 1)
    else:
        ticks = np.linspace(math.ceil(min_rt / 10.0) * 10,
                             math.ceil(max_rt / 10.0) * 10 - 10,
                             math.ceil(max_rt / 10.0) - math.ceil(min_rt / 10.0))
    currenttick = 0
    for rt_val in rt_values:
        if currenttick < len(ticks) and rt_val > ticks[currenttick]:
            position = int((rt_val - min_rt) / (max_rt - min_rt) * x_plot.max())
            if position < len(x_plot):
                ticksx.append(x_plot[position])
                tickslabels.append(ticks[currenttick])
            currenttick += 1
    plt.xticks(ticksx, tickslabels, fontsize=12)

    ax.set_xlabel('Retention time (min)', fontsize=14)
    ax.set_ylabel(f"Covariance with \n signal at {target:.2f} min", fontsize=14)
    ax.set_title(f'STOCSY from signal at {target:.2f} min ({mode} model)', fontsize=16)

    # Configurar função hover para visualização interativa
    text = ax.text(1, 1, '')
    lnx = plt.plot([60, 60], [0, 1.5], color='black', linewidth=0.3)
    lny = plt.plot([0, 100], [1.5, 1.5], color='black', linewidth=0.3)
    lnx[0].set_linestyle('None')
    lny[0].set_linestyle('None')

    def hover(event):
        if event.inaxes == ax:
            inv = ax.transData.inverted()
            maxcoord = ax.transData.transform((x_plot[0], 0))[0]
            mincoord = ax.transData.transform((x_plot[-1], 0))[0]
            rt_val = ((maxcoord - mincoord) - (event.x - mincoord)) / (maxcoord - mincoord) * (max_rt - min_rt) + min_rt
            i = int(((maxcoord - mincoord) - (event.x - mincoord)) / (maxcoord - mincoord) * len(covar))
            if 0 <= i < len(covar):
                cov_val = covar[i]
                cor_val = corr[i]
                text.set_visible(True)
                text.set_position((event.xdata, event.ydata))
                text.set_text(f'{rt_val:.2f} min, covariance: {cov_val:.6f}, correlation: {cor_val:.2f}')
                lnx[0].set_data([event.xdata, event.xdata], [-1, 1])
                lnx[0].set_linestyle('--')
                lny[0].set_data([x_plot[0], x_plot[-1]], [cov_val, cov_val])
                lny[0].set_linestyle('--')
        else:
            text.set_visible(False)
            lnx[0].set_linestyle('None')
            lny[0].set_linestyle('None')
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    if not os.path.exists('images'):
        os.mkdir('images')
    plt.savefig(f"images/stocsy_from_{target}min_{mode}.pdf", transparent=True, dpi=300)

    import mpld3
    html_str = mpld3.fig_to_html(fig)
    with open(f"images/stocsy_interactive_{target}min_{mode}.html", "w") as f:
        f.write(html_str)

    plt.show()
    return corr, covar


def run_parafac_uv_analysis(normalized_scaled_df, rank=3, init='svd', n_iter_max=1000, tol=1e-6, output_dir="images"):
    """
    Perform PARAFAC analysis on UV data and save plots.

    Parameters:
    - normalized_scaled_df: DataFrame with first column = Retention Time, other columns = wavelength intensities (columns can have 'nm' suffix).
    - rank: Number of components to extract.
    - init: Initialization method ('svd', 'random', etc.).
    - n_iter_max: Maximum number of iterations.
    - tol: Convergence tolerance.
    - output_dir: Directory to save plots.

    Returns:
    - weights: PARAFAC weights.
    - factor_matrices: Tuple of factor matrices (retention_time_factors, wavelength_factors, ...).
    """
    # Prepare data
    retention_times = normalized_scaled_df.iloc[:, 0].values

    # FIX: Remove 'nm' from column names before converting to float (in case they're labeled like '190.64nm')
    wavelengths = normalized_scaled_df.columns[1:].str.replace('nm', '', regex=False).astype(float)

    wavelength_data = normalized_scaled_df.iloc[:, 1:].values

    # Perform PARAFAC decomposition
    tensor_data = tensor(wavelength_data)
    weights, factor_matrices = parafac(
        tensor_data,
        rank=rank,
        init=init,
        n_iter_max=n_iter_max,
        tol=tol
    )

    retention_time_factors, wavelength_factors = factor_matrices[:2]

    # Interpolate wavelength factors to match the full wavelength range
    interpolated_factors = []
    original_wavelengths = np.linspace(wavelengths.min(), wavelengths.max(), wavelength_factors.shape[0])
    for i in range(wavelength_factors.shape[1]):
        interp_func = interp1d(original_wavelengths, wavelength_factors[:, i], kind='linear', fill_value="extrapolate")
        interpolated_factors.append(interp_func(wavelengths))
    interpolated_factors = np.array(interpolated_factors).T

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot Retention Time Factors (interactive)
    fig = go.Figure()
    for i in range(rank):
        fig.add_trace(go.Scatter(
            x=retention_times,
            y=retention_time_factors[:, i],
            mode='lines',
            name=f'Component {i + 1}'
        ))
    fig.update_layout(
        title='Retention Time Factors',
        xaxis_title='Retention Time (min)',
        yaxis_title='Factor Value',
        hovermode='x unified',
        template='plotly_white'
    )
    html_output = os.path.join(output_dir, 'retention_time_factors.html')
    fig.write_html(html_output)
    print(f"Retention Time Factors plot saved to: {html_output}")

    # Plot Wavelength Factors (interactive)
    fig2 = go.Figure()
    for i in range(rank):
        fig2.add_trace(go.Scatter(
            x=wavelengths,
            y=interpolated_factors[:, i],
            mode='lines',
            name=f'Component {i + 1}'
        ))
    fig2.update_layout(
        title='Wavelength Factors',
        xaxis_title='Wavelength (nm)',
        yaxis_title='Factor Value',
        hovermode='x unified',
        template='plotly_white'
    )
    html_output2 = os.path.join(output_dir, 'wavelength_factors.html')
    fig2.write_html(html_output2)
    print(f"Wavelength Factors plot saved to: {html_output2}")

    return weights, (retention_time_factors, interpolated_factors)



# --------------------------------------------------------------------------
#                    10) Data-Export
# --------------------------------------------------------------------------

import os
import csv
import re

def sanitize_string(s):
    """
    Sanitize a string so that it contains only English letters, numbers, and underscores.
    """
    return re.sub(r'[^a-zA-Z0-9_]', '', s)

def export_metaboanalyst(aligned_df, df_metadata,
                         sample_id_col="LC_filename",
                         class_col="ATTRIBUTE_classification",
                         output_file="metaboanalyst_input.csv"):
    """
    Export NMR data to a CSV suitable for MetaboAnalyst with the following format:
      - First row: column headers (e.g., "RT (min)" plus sample IDs)
      - Second row: blank under "RT (min)", then each sample's classification
      - Remaining rows: the actual data from aligned_df

    The function filters the metadata to include only samples present in the data.
    It also sanitizes sample IDs and classification values so that only English letters,
    numbers, and underscores remain.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        NMR data with the first column as the retention time (e.g., "RT (ppm)" or "RT (min)")
        and subsequent columns as sample intensities.
        (If there is a column "Chemical Shift (ppm)", it will be ignored.)
    df_metadata : pd.DataFrame
        Metadata DataFrame containing sample information. Must include:
          - sample_id_col: the column with sample IDs.
          - class_col: the column with class/group information for each sample.
    sample_id_col : str, optional
        The column in df_metadata that contains sample IDs.
    class_col : str, optional
        The column in df_metadata that contains the sample classification.
    output_file : str, optional
        The output CSV filename.

    Returns
    -------
    new_df : pd.DataFrame
        The new DataFrame (with only valid sample columns) that is saved to CSV.
    """
    import os
    import csv
    import re
    
    def sanitize_string(s):
        """Sanitize a string to contain only letters, numbers, and underscores."""
        return re.sub(r'[^a-zA-Z0-9_]', '', str(s))
    
    # --- Step 1: Sanitize column headers in aligned_df ---
    # If there's a column named "Chemical Shift (ppm)", drop it.
    if "Chemical Shift (ppm)" in aligned_df.columns:
        aligned_df = aligned_df.drop(columns=["Chemical Shift (ppm)"])
    
    orig_col_names = list(aligned_df.columns)
    # Assume the first column is the retention time column (keep it unchanged).
    # For all other columns (assumed to be sample IDs), sanitize their names.
    sanitized_sample_cols = [sanitize_string(s) for s in orig_col_names[1:]]
    # Preserve the first column header as-is.
    sanitized_col_names = [orig_col_names[0]] + sanitized_sample_cols
    aligned_df = aligned_df.copy()
    aligned_df.columns = sanitized_col_names

    # --- Step 2: Identify sample columns ---
    sample_cols = sanitized_col_names[1:]
    
    # --- Step 3: Filter and sanitize metadata ---
    meta = df_metadata.copy()
    meta[sample_id_col] = meta[sample_id_col].apply(sanitize_string)
    meta[class_col] = meta[class_col].apply(sanitize_string)
    meta_indexed = meta.set_index(sample_id_col)
    classification_series = meta_indexed.reindex(sample_cols)[class_col]
    
    valid_sample_cols = classification_series.dropna().index.tolist()
    if len(valid_sample_cols) < len(sample_cols):
        missing = set(sample_cols) - set(valid_sample_cols)
        print(f"Warning: The following sample IDs are missing metadata and will be excluded: {missing}")
    
    # --- Step 4: Build the classification row ---
    classification_row = [""]
    classification_row.extend(classification_series.loc[valid_sample_cols].values)
    
    # --- Step 5: Build a new DataFrame using only the retention time column and valid sample columns ---
    # Use "RT (min)" if it exists; otherwise, assume the first column is the RT column.
    if "RT (min)" in aligned_df.columns:
        rt_col = "RT (min)"
    else:
        rt_col = aligned_df.columns[0]
    new_df = aligned_df[[rt_col] + valid_sample_cols].copy()
    
    # --- Step 6: Write the CSV with the classification row after the header ---
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_df.columns.tolist())
        writer.writerow(classification_row)
        for i in range(len(new_df)):
            writer.writerow(new_df.iloc[i].values)
    
    print(f"MetaboAnalyst input CSV saved as: {output_file}")
    return new_df


# --------------------------------------------------------------------------
#                    11) Data-Processing Report
# --------------------------------------------------------------------------
def print_data_processing_report(start_rt, end_rt, samples_to_remove,
                                 aligned_method, 
                                 transformation_method, 
                                 normalization_method, 
                                 scale_method):
    """
    Print a concise data-processing report for LC analysis.
    """
    samples_str = ", ".join(samples_to_remove) if samples_to_remove else "None"
    report = f"""
Data-Processing Overview
----------------------
1. Unwanted outer RT regions removed: {start_rt}–{end_rt} min
2. Removed samples: {samples_str}
3. Alignment method: {aligned_method}
4. Transformation method: {transformation_method}
5. Normalization method: {normalization_method}
6. Scaling method: {scale_method}
    """
    print(report.strip())

# --------------------------------------------------------------------------
#                       End of data_processing_NMR.py
# --------------------------------------------------------------------------