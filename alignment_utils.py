# alignment_utils.py
# Drop-in alignment helpers for the pyMetaFlow Streamlit app.
# Tries to use data_processing_HPLC.py (your original functions). If not importable,
# falls back to embedded implementations.
from __future__ import annotations

import numpy as np
import pandas as pd

# Optional Streamlit imports for the small UI helper.
try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

# Try to import your module with PAFFT/RAFFT/iCOshift implementations
_dp = None
try:
    import data_processing_HPLC as _dp  # your file
except Exception:
    _dp = None

# ---- Fallback implementations (from your data_processing_HPLC.py) ----
def _fft_corr(spectrum, target, shift):
    M = len(target)
    diff = 1_000_000
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

def _move_seg(seg, lag):
    if lag == 0 or lag >= len(seg):
        return seg
    if lag > 0:
        ins = np.full(lag, seg[0])
        return np.concatenate([ins, seg[:-lag]])
    else:
        lag_abs = abs(lag)
        ins = np.full(lag_abs, seg[-1])
        return np.concatenate([seg[lag_abs:], ins])

def _find_mid(spec):
    M = int(np.ceil(len(spec) / 2))
    offset = int(np.floor(M / 4))
    start = max(M - offset, 0)
    end = min(M + offset, len(spec))
    spec_segment = spec[start:end]
    I = np.argmin(spec_segment)
    mid = I + start
    return mid

def _recur_align(spectrum, reference, shift, lookahead):
    if len(spectrum) < 10:
        return spectrum
    lag = _fft_corr(spectrum, reference, shift)
    if lag == 0 and lookahead <= 0:
        return spectrum
    else:
        if lag == 0:
            lookahead -= 1
        if abs(lag) < len(spectrum):
            aligned = _move_seg(spectrum, lag)
        else:
            aligned = spectrum.copy()
        mid = _find_mid(aligned)
        first_seg = _recur_align(aligned[:mid], reference[:mid], shift, lookahead)
        second_seg = _recur_align(aligned[mid:], reference[mid:], shift, lookahead)
        return np.concatenate([first_seg, second_seg])

def _RAFFT_df(data: pd.DataFrame, reference_idx: int = 0, shift_RT: float | None = None, lookahead: int = 1):
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
        aligned_intensities[i, :] = _recur_align(intensities[i, :], reference_spectrum, shift, lookahead)
    aligned_intensities = aligned_intensities.T
    aligned_df = pd.DataFrame(np.column_stack((axis, aligned_intensities)), columns=data.columns)
    return aligned_df

def _find_min(samseg, refseg):
    Cs = np.sort(samseg); Is = np.argsort(samseg)
    Cr = np.sort(refseg); Ir = np.argsort(refseg)
    n_limit = max(1, int(round(len(Cs) / 20)))
    for i in range(n_limit):
        for j in range(n_limit):
            if Ir[j] == Is[i]:
                return Is[i]
    return Is[0]

def _PAFFT(spectrum, reference, segSize, shift):
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
            minpos = _find_min(samseg, refseg)
            endpos = startpos + minpos + segSize
            samseg = spectrum[startpos:endpos]
            refseg = reference[startpos:endpos]
        lag = _fft_corr(samseg, refseg, shift)
        moved = _move_seg(samseg, lag)
        aligned_segments.append(moved)
        startpos = endpos + 1
    aligned_full = np.concatenate(aligned_segments)
    if len(aligned_full) < n_points:
        aligned_full = np.pad(aligned_full, (0, n_points - len(aligned_full)), mode='edge')
    else:
        aligned_full = aligned_full[:n_points]
    return aligned_full

def _PAFFT_df(data: pd.DataFrame, segSize_RT: float, reference_idx: int = 0, shift_RT: float | None = None):
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
        aligned_intensities[i, :] = _PAFFT(intensities[i, :], reference_spectrum, segSize, shift)
    aligned_intensities = aligned_intensities.T
    aligned_df = pd.DataFrame(np.column_stack((axis, aligned_intensities)), columns=data.columns)
    return aligned_df
# ---- End fallbacks ----

def align_df(df: pd.DataFrame, method: str, **kwargs) -> pd.DataFrame:
    """
    Run alignment using one of: 'Icoshift', 'PAFFT', 'RAFFT', or 'None'.
    kwargs per method:
      - Icoshift: n_intervals:int=50, target:str='maxcorr'
      - PAFFT: segSize_RT:float, reference_idx:int=0, shift_RT:float|None=None
      - RAFFT: reference_idx:int=0, shift_RT:float|None=None, lookahead:int=1
    """
    if method == "None":
        return df

    if method == "Icoshift":
        n_intervals = int(kwargs.get("n_intervals", 50))
        target = kwargs.get("target", "maxcorr")
        if _dp and hasattr(_dp, "align_samples_using_icoshift"):
            return _dp.align_samples_using_icoshift(df, n_intervals=n_intervals, target=target)
        # No good fallback here without pyicoshift; return df unchanged as a safeguard
        return df

    if method == "PAFFT":
        reference_idx = int(kwargs.get("reference_idx", 0))
        segSize_RT = float(kwargs.get("segSize_RT", 0.2))
        shift_RT = kwargs.get("shift_RT", None)
        if isinstance(shift_RT, (int, float)) and float(shift_RT) <= 0:
            shift_RT = None
        if _dp and hasattr(_dp, "PAFFT_df"):
            return _dp.PAFFT_df(df, segSize_RT=segSize_RT, reference_idx=reference_idx, shift_RT=shift_RT)
        return _PAFFT_df(df, segSize_RT=segSize_RT, reference_idx=reference_idx, shift_RT=shift_RT)

    if method == "RAFFT":
        reference_idx = int(kwargs.get("reference_idx", 0))
        lookahead = int(kwargs.get("lookahead", 1))
        shift_RT = kwargs.get("shift_RT", None)
        if isinstance(shift_RT, (int, float)) and float(shift_RT) <= 0:
            shift_RT = None
        if _dp and hasattr(_dp, "RAFFT_df"):
            return _dp.RAFFT_df(df, reference_idx=reference_idx, shift_RT=shift_RT, lookahead=lookahead)
        return _RAFFT_df(df, reference_idx=reference_idx, shift_RT=shift_RT, lookahead=lookahead)

    raise ValueError(f"Unknown alignment method: {method}")

def alignment_controls(df: pd.DataFrame, sample_names: list[str] | None = None):
    """
    Small Streamlit UI for picking alignment method + params.
    Returns: (method, params_dict)
    """
    if st is None:
        raise RuntimeError("alignment_controls requires Streamlit")

    method = st.selectbox("Alignment method", ["None", "Icoshift", "PAFFT", "RAFFT"], index=1)

    params = {}
    if method == "Icoshift":
        params["n_intervals"] = st.number_input("Icoshift: number of intervals", min_value=1, max_value=500, value=50, step=1)
        params["target"] = st.selectbox("Icoshift target", ["maxcorr", "median", "first"], index=0)

    elif method in ("PAFFT", "RAFFT"):
        # Reference selection
        if sample_names is None:
            sample_names = list(df.columns[1:])
        ref_label = st.selectbox("Reference sample", sample_names, index=0)
        reference_idx = sample_names.index(ref_label)
        params["reference_idx"] = reference_idx

        if method == "PAFFT":
            params["segSize_RT"] = st.number_input("Segment size (min)", min_value=0.01, max_value=10.0, value=0.20, step=0.01, format="%.2f")
            shift_rt = st.number_input("Max shift (min) [0 = auto]", min_value=0.0, max_value=5.0, value=0.0, step=0.05, format="%.2f")
            params["shift_RT"] = None if shift_rt <= 0 else float(shift_rt)
        else:
            params["lookahead"] = st.number_input("RAFFT lookahead depth", min_value=0, max_value=5, value=1, step=1)
            shift_rt = st.number_input("Max shift (min) [0 = auto]", min_value=0.0, max_value=5.0, value=0.0, step=0.05, format="%.2f")
            params["shift_RT"] = None if shift_rt <= 0 else float(shift_rt)

    return method, params
