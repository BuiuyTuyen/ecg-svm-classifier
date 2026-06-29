"""
Shared ECG feature extraction — imported by KNN.py, SVM.py, RF.py, and hardware code.
Single source of truth: all four callers extract the same 23 features.
"""
import numpy as np
import pandas as pd

FS_TARGET = 360  # common target sampling rate (matches MIT-BIH)


def extract_features(beat, pre_rr_sec, post_rr_sec, fs=FS_TARGET):
    """
    23 features: morphological + statistical + energy + RR.

    beat        : numpy array of exactly 2*HALF_WIN samples at `fs` Hz
    pre_rr_sec  : RR interval before this beat, in seconds
    post_rr_sec : RR interval after this beat, in seconds
    fs          : sampling rate (default FS_TARGET=360)

    Returns a plain Python list of 23 floats in this order:
      r_val, q_val, s_val, r_val-q_val, r_val-s_val,
      qrs, pre_r, mean, std, skew, kurt, rms,
      e1, e2, e3, e4, (e1+e2)/(e1+e2+e3+e4),
      area, zero_crossings, t_prox,
      pre_rr_samples, post_rr_samples, rr_ratio
    """
    n = len(beat); mid = n // 2; q4 = n // 4
    r_val = beat[mid]
    q_idx = int(np.argmin(beat[:mid]));        q_val = beat[q_idx]
    s_idx = int(np.argmin(beat[mid:])) + mid;  s_val = beat[s_idx]
    qrs   = s_idx - q_idx
    pre_r = mid - q_idx
    mean_v = beat.mean()
    std_v  = beat.std()
    skew_v = float(pd.Series(beat).skew())
    kurt_v = float(pd.Series(beat).kurtosis())
    rms_v  = float(np.sqrt(np.mean(beat ** 2)))
    e1 = float(np.sum(beat[:q4]       ** 2))
    e2 = float(np.sum(beat[q4:2*q4]   ** 2))
    e3 = float(np.sum(beat[2*q4:3*q4] ** 2))
    e4 = float(np.sum(beat[3*q4:]     ** 2))
    tot  = e1 + e2 + e3 + e4 + 1e-12
    area = float(np.trapezoid(np.abs(beat)))
    zc   = int(np.sum(np.diff(np.sign(beat)) != 0))
    t_prx = float(beat[-30:].mean())
    pre_rr_samples  = pre_rr_sec  * fs
    post_rr_samples = post_rr_sec * fs
    rr_ratio = pre_rr_samples / (post_rr_samples + 1e-8)
    return [r_val, q_val, s_val, r_val - q_val, r_val - s_val,
            float(qrs), float(pre_r), mean_v, std_v, skew_v, kurt_v, rms_v,
            e1, e2, e3, e4, (e1 + e2) / tot, area, float(zc), t_prx,
            float(pre_rr_samples), float(post_rr_samples), float(rr_ratio)]
