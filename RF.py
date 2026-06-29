"""
ECG Random Forest Classifier v6 — MIT-BIH + svdb, AAMI 3-class
================================================================
Changes from v5 (review corrections):
  - [FIX] Added PCA (95% variance) after scaling — RF now uses the same
    PCA-reduced feature space as KNN and SVM, enabling a fair comparison
  - [FIX] SMOTEENN now applied to PCA-reduced space (was full 23-D),
    consistent with KNN and SVM pipelines
  - [FIX] GridSearch changed from 3-fold to 5-fold CV — consistent with
    KNN (was 3-fold, KNN was 5-fold; inconsistent experimental conditions)

Changes from v4 (v5):
  - 3-class AAMI EC57: N, S, V (F merged into V, Q excluded)
  - Added MIT-BIH Supraventricular Arrhythmia Database (svdb, 78 records)
    resampled from 128 Hz to 360 Hz to match MIT-BIH
  - Excluded paced records 102, 104, 107, 217 (Chazal et al. 2004)
  - Per-class precision/recall/F1 saved to JSON for thesis table reuse
"""
import warnings
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from collections import Counter

# ════════════════════════════════════════════════════════════
# TIMING UTILITY
# ════════════════════════════════════════════════════════════
timings = {}

class Timer:
    def __init__(self, label): self.label = label
    def __enter__(self): self.t0 = time.perf_counter(); return self
    def __exit__(self, *args):
        dt = time.perf_counter() - self.t0
        timings[self.label] = dt
        print(f"  ⏱ {self.label}: {dt:.2f} s ({dt/60:.2f} min)")

def fmt_hms(sec):
    m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from imblearn.combine import SMOTEENN
from scipy.signal import resample
import wfdb
from ecg_features import extract_features

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
FS_TARGET       = 360                    # Common sampling rate after resampling
HALF_WIN_TARGET = 90                     # 90 samples at 360 Hz = 0.25 s
WINDOW_TARGET   = 2 * HALF_WIN_TARGET    # 180 samples = 0.5 s

# AAMI EC57 3-class scheme (N, S, V)
CLASSES = ["N", "S", "V"]
BEAT_MAP = {
    # N (Normal)
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # S (Supraventricular ectopic)
    "S": "S", "A": "S", "a": "S", "J": "S",
    # V (Ventricular ectopic) — F (fusion) merged in per AAMI recommendation
    "V": "V", "E": "V", "F": "V",
    # Q and f excluded: unclassifiable beats not in AAMI EC57 evaluation classes
}

TEST_SIZE     = 0.20
SEED          = 42
OUTPUT_DIR    = "./outputs"
MODEL_PATH    = f"{OUTPUT_DIR}/ecg_rf_3class_v6.pkl"
RESULTS_JSON  = f"{OUTPUT_DIR}/results_rf_3class_v6.json"

# MIT-BIH records — exclude paced patients (Chazal et al. 2004)
PACED_RECORDS = {"102", "104", "107", "217"}
MITDB_ALL = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
MITDB_RECORDS = [r for r in MITDB_ALL if r not in PACED_RECORDS]  # 44 records

# svdb records — 78 records
SVDB_RECORDS = [
    "800", "801", "802", "803", "804", "805", "806", "807", "808", "809",
    "810", "811", "812", "820", "821", "822", "823", "824", "825", "826",
    "827", "828", "829", "840", "841", "842", "843", "844", "845", "846",
    "847", "848", "849", "850", "851", "852", "853", "854", "855", "856",
    "857", "858", "859", "860", "861", "862", "863", "864", "865", "866",
    "867", "868", "869", "870", "871", "872", "873", "874", "875", "876",
    "877", "878", "879", "880", "881", "882", "883", "884", "885", "886",
    "887", "888", "889", "890", "891", "892", "893", "894",
]

LOCAL_MITDB_PATH = r"D:\machine learning\mitdb_data\mitdb"
LOCAL_SVDB_PATH  = r"D:\machine learning\svdb_data\svdb"

# ════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING (MIT-BIH + svdb)
# ════════════════════════════════════════════════════════════
print("═" * 60)
print(" STEP 1 — DATA LOADING (MIT-BIH + svdb, AAMI 3-class: N, S, V)")
print("═" * 60)

if not os.path.isdir(LOCAL_MITDB_PATH):
    raise FileNotFoundError(
        f"MIT-BIH folder not found: {LOCAL_MITDB_PATH}\n"
        "Verify the path and that .dat/.hea/.atr files are present."
    )


def ensure_svdb_local():
    """Download svdb to local path on first run (~50 MB, one-time)."""
    hea_count = 0
    if os.path.isdir(LOCAL_SVDB_PATH):
        hea_count = sum(1 for f in os.listdir(LOCAL_SVDB_PATH) if f.endswith(".hea"))
    if hea_count >= 70:
        print(f"  svdb found locally ({hea_count} .hea files) at {LOCAL_SVDB_PATH}")
        return
    print(f"  svdb not found locally — downloading from PhysioNet to {LOCAL_SVDB_PATH}")
    print("  (~50 MB, one-time download, ~3-5 min on a typical connection)")
    os.makedirs(LOCAL_SVDB_PATH, exist_ok=True)
    wfdb.dl_database("svdb", dl_dir=LOCAL_SVDB_PATH)
    print("  svdb download complete")


def load_record(rec, db_path, fs_native):
    """
    Load one record, extract beats around annotated R-peaks, resample to FS_TARGET.
    Returns lists of (beat_window, label, pre_rr_seconds, post_rr_seconds).
    """
    record_path = os.path.join(db_path, rec)
    record = wfdb.rdrecord(record_path)
    ann    = wfdb.rdann(record_path, "atr")

    sig     = record.p_signal[:, 0]      # use channel 0 (MLII for MIT-BIH, ECG1 for svdb)
    half_win_native = int(round(0.25 * fs_native))   # 0.25 s window in native samples

    # Pre-filter to beat annotations only so RR intervals are beat-to-beat,
    # not contaminated by rhythm/signal-quality annotations ("+", "~", "|", etc.)
    beat_pairs   = [(s, sym) for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_MAP]
    if not beat_pairs:
        return [], [], [], []
    beat_samples = [s for s, _ in beat_pairs]
    beat_symbols = [sym for _, sym in beat_pairs]

    beats, labels, pre_rrs, post_rrs = [], [], [], []

    for i, (idx, sym) in enumerate(zip(beat_samples, beat_symbols)):
        s, e = idx - half_win_native, idx + half_win_native
        if s < 0 or e > len(sig):
            continue

        beat_native = sig[s:e]
        # Resample to FS_TARGET (no-op if fs_native == FS_TARGET)
        if fs_native != FS_TARGET:
            beat = resample(beat_native, WINDOW_TARGET)
        else:
            beat = beat_native
        beats.append(beat)
        labels.append(BEAT_MAP[sym])

        # RR intervals in SECONDS (beat-to-beat only)
        if i > 0:
            pre_rr_sec = (beat_samples[i] - beat_samples[i-1]) / fs_native
        else:
            pre_rr_sec = (beat_samples[1] - beat_samples[0]) / fs_native if len(beat_samples) > 1 else 1.0
        if i < len(beat_samples) - 1:
            post_rr_sec = (beat_samples[i+1] - beat_samples[i]) / fs_native
        else:
            post_rr_sec = pre_rr_sec
        pre_rrs.append(pre_rr_sec)
        post_rrs.append(post_rr_sec)

    return beats, labels, pre_rrs, post_rrs


with Timer("data_loading_total"):
    # --- Load MIT-BIH (44 records, paced excluded) ---
    with Timer("load_mitdb"):
        print(f"\n Loading MIT-BIH Arrhythmia ({len(MITDB_RECORDS)} records, paced excluded)...")
        mitdb_beats, mitdb_labels, mitdb_pre, mitdb_post, mitdb_recs = [], [], [], [], []
        for rec in MITDB_RECORDS:
            try:
                b, l, pre, post = load_record(rec, LOCAL_MITDB_PATH, fs_native=360)
                mitdb_beats.extend(b)
                mitdb_labels.extend(l)
                mitdb_pre.extend(pre)
                mitdb_post.extend(post)
                mitdb_recs.extend([f"mitdb_{rec}"] * len(b))
                print(f"  MIT-BIH {rec}: {len(b)} beats")
            except FileNotFoundError:
                print(f"  MIT-BIH {rec}: file not found — skipped")
        print(f"\n MIT-BIH subtotal: {len(mitdb_beats)} beats")
        print(f"   Class distribution: {dict(sorted(Counter(mitdb_labels).items()))}")

    # --- Load svdb (78 records) ---
    ensure_svdb_local()
    with Timer("load_svdb"):
        print(f"\n Loading MIT-BIH Supraventricular ({len(SVDB_RECORDS)} records, 128→360 Hz)...")
        svdb_beats, svdb_labels, svdb_pre, svdb_post, svdb_recs = [], [], [], [], []
        for rec in SVDB_RECORDS:
            try:
                b, l, pre, post = load_record(rec, LOCAL_SVDB_PATH, fs_native=128)
                svdb_beats.extend(b)
                svdb_labels.extend(l)
                svdb_pre.extend(pre)
                svdb_post.extend(post)
                svdb_recs.extend([f"svdb_{rec}"] * len(b))
                print(f"  svdb {rec}: {len(b)} beats")
            except FileNotFoundError:
                print(f"  svdb {rec}: file not found — skipped")
        print(f"\n svdb subtotal: {len(svdb_beats)} beats")
        print(f"   Class distribution: {dict(sorted(Counter(svdb_labels).items()))}")

# --- Combine ---
all_beats  = mitdb_beats + svdb_beats
all_labels = mitdb_labels + svdb_labels
all_pre    = mitdb_pre + svdb_pre
all_post   = mitdb_post + svdb_post
all_recs   = mitdb_recs + svdb_recs   # per-beat record ID for patient-wise split

print(f"\n COMBINED TOTAL: {len(all_beats)} beats")
print(f" Class distribution: {dict(sorted(Counter(all_labels).items()))}")

# ════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESSING (outlier removal)
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(" STEP 2 — PREPROCESSING (IQR outlier removal)")
print("═" * 60)

X_all    = np.array(all_beats, dtype=float)
y_all    = np.array(all_labels)
pre_all  = np.array(all_pre,   dtype=float)
post_all = np.array(all_post,  dtype=float)
recs_all = np.array(all_recs)

with Timer("preprocessing"):
    amp    = X_all.max(axis=1) - X_all.min(axis=1)   # kept for post-split outlier removal
    print(f" Beats loaded: {len(y_all)}")
    print(f" Class distribution: {dict(sorted(Counter(y_all).items()))}")

# ════════════════════════════════════════════════════════════
# STEP 3 — FEATURE EXTRACTION (unchanged from v4)
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(" STEP 3 — FEATURE EXTRACTION (23 features)")
print("═" * 60)


with Timer("feature_extraction"):
    print(" Extracting features…")
    X_feat = np.nan_to_num(np.array([
        extract_features(b, pr, po)
        for b, pr, po in zip(X_all, pre_all, post_all)
    ]))
    print(f" Feature matrix shape: {X_feat.shape}")

# ════════════════════════════════════════════════════════════
# STEP 4 — PATIENT-WISE 80/20 SPLIT
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(" STEP 4 — PATIENT-WISE 80/20 SPLIT (per database)")
print("═" * 60)

le    = LabelEncoder()
y_enc = le.fit_transform(y_all)

# Split records (not beats) per database so no patient leaks between train/test
mitdb_ids = [f"mitdb_{r}" for r in MITDB_RECORDS]
svdb_ids  = [f"svdb_{r}"  for r in SVDB_RECORDS]
m_train, m_test = train_test_split(mitdb_ids, test_size=TEST_SIZE, random_state=SEED)
s_train, s_test = train_test_split(svdb_ids,  test_size=TEST_SIZE, random_state=SEED)
train_recs = set(m_train) | set(s_train)
test_recs  = set(m_test)  | set(s_test)

train_mask = np.isin(recs_all, list(train_recs))
test_mask  = np.isin(recs_all, list(test_recs))
X_train, X_test = X_feat[train_mask], X_feat[test_mask]
y_train, y_test = y_enc[train_mask],  y_enc[test_mask]

print(f" Classes      : {list(le.classes_)}")
print(f" Train records: {len(train_recs)} ({len(m_train)} mitdb + {len(s_train)} svdb)")
print(f" Test  records: {len(test_recs)}  ({len(m_test)}  mitdb + {len(s_test)}  svdb)")
print(f" Train beats  : {len(y_train)}  {dict(zip(le.classes_, np.bincount(y_train, minlength=len(le.classes_))))}")
print(f" Test  beats  : {len(y_test)}   {dict(zip(le.classes_, np.bincount(y_test,  minlength=len(le.classes_))))}")

# Fix #1 — warn if any class has 0 train/test beats (patient-wise split may drop rare classes)
missing_test  = [le.classes_[i] for i in range(len(le.classes_)) if (y_test  == i).sum() == 0]
missing_train = [le.classes_[i] for i in range(len(le.classes_)) if (y_train == i).sum() == 0]
if missing_train:
    print(f" ❌ ERROR: Classes with 0 train beats: {missing_train} — model CANNOT learn these")
if missing_test:
    print(f" ⚠ WARNING: Classes with 0 test beats: {missing_test} — metrics undefined for these")

# Outlier removal using TRAIN amplitude stats only (fix: no test-set leakage)
amp_train  = amp[train_mask]
amp_test   = amp[test_mask]
Q1, Q3     = np.percentile(amp_train, 25), np.percentile(amp_train, 75)
iqr        = Q3 - Q1
keep_train = (amp_train >= Q1 - 3*iqr) & (amp_train <= Q3 + 3*iqr)
keep_test  = (amp_test  >= Q1 - 3*iqr) & (amp_test  <= Q3 + 3*iqr)
X_train, y_train = X_train[keep_train], y_train[keep_train]
X_test,  y_test  = X_test[keep_test],   y_test[keep_test]
print(f" After outlier removal: {len(y_train)} train / {len(y_test)} test beats")

# ════════════════════════════════════════════════════════════
# STEP 5 — SCALING
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(" STEP 5 — SCALING")
print("═" * 60)

with Timer("scaling_pca"):
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    # PCA added (v6): RF now uses the same reduced feature space as KNN/SVM
    # so that model comparisons are on equal footing.
    pca      = PCA(n_components=0.95, random_state=SEED)
    X_tr_pca = pca.fit_transform(X_train_sc)
    X_te_pca = pca.transform(X_test_sc)
    print(f" PCA: {X_train.shape[1]} → {X_tr_pca.shape[1]} components "
          f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance retained)")

# ════════════════════════════════════════════════════════════
# STEP 6 — SMOTEENN BALANCING
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(" STEP 6 — SMOTEENN BALANCING (train set only)")
print("═" * 60)
print(" This step may take 5–15 minutes on a large combined dataset…")

with Timer("smoteenn"):
    smoteenn = SMOTEENN(random_state=SEED)
    # v6: balance in PCA space (was full 23-D in v5), consistent with KNN/SVM
    X_train_bal, y_train_bal = smoteenn.fit_resample(X_tr_pca, y_train)
    print(f" After balancing: {dict(zip(le.classes_, np.bincount(y_train_bal)))}")

# ════════════════════════════════════════════════════════════
# STEP 7 — GRIDSEARCH (5-fold CV)
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(" STEP 7 — RANDOM FOREST GRIDSEARCH (5-fold CV)")
print("═" * 60)

param_grid = {
    "n_estimators":      [100, 200],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5],
}
print(f" Grid: {sum(len(v) for v in param_grid.values())} hyperparameters, "
      f"{2*3*2} combinations × 5-fold = {2*3*2*5} fits")

with Timer("gridsearch_rf"):
    # v6: changed from 3-fold to 5-fold to match KNN — consistent experimental conditions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid_search = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2,
    )
    grid_search.fit(X_train_bal, y_train_bal)
    best_params = grid_search.best_params_
    print(f"\n Best params      : {best_params}")
    print(f" Best CV f1_macro : {grid_search.best_score_:.4f}")

# ════════════════════════════════════════════════════════════
# STEP 8 — FINAL MODEL + EVALUATION
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(" STEP 8 — FINAL MODEL + TEST SET EVALUATION")
print("═" * 60)

with Timer("final_training"):
    rf_final = RandomForestClassifier(
        n_estimators      = best_params["n_estimators"],
        max_depth         = best_params["max_depth"],
        min_samples_split = best_params["min_samples_split"],
        class_weight      = "balanced",
        random_state      = SEED,
        n_jobs            = -1,
    )
    rf_final.fit(X_train_bal, y_train_bal)

os.makedirs(OUTPUT_DIR, exist_ok=True)

joblib.dump({
    "scaler":      scaler,
    "pca":         pca,          # v6: RF now includes PCA step
    "rf":          rf_final,
    "label_enc":   le,
    "best_params": best_params,
    "window":      WINDOW_TARGET,
    "fs":          FS_TARGET,
    "classes":     list(le.classes_),
    "test_size":   TEST_SIZE,
    "has_rr":      True,
    "schema":      "AAMI_3class_v6",
}, MODEL_PATH)
print(f"\n Model saved : {MODEL_PATH}")

with Timer("prediction_total"):
    y_pred = rf_final.predict(X_te_pca)   # v6: use PCA-transformed test features
n_test = len(y_test)
per_beat_ms = timings["prediction_total"] / n_test * 1000.0

# Single-beat latency benchmark (real-time inference)
single_beat = X_te_pca[:1]   # v6: PCA space
n_single_runs = 1000
t0 = time.perf_counter()
for _ in range(n_single_runs):
    _ = rf_final.predict(single_beat)
single_latency_ms = (time.perf_counter() - t0) / n_single_runs * 1000.0
timings["single_beat_latency_ms"] = single_latency_ms

acc    = accuracy_score(y_test, y_pred)
print("\n" + "═" * 60)
print(" FINAL RESULTS ON 20% TEST SET")
print("═" * 60)
print(f" Accuracy             : {acc * 100:.2f}%")
print(f" Total predict time   : {timings['prediction_total']:.3f} s for {n_test} beats")
print(f" Avg latency / beat   : {per_beat_ms:.3f} ms")
print(f" Single-beat latency  : {single_latency_ms:.3f} ms (avg of {n_single_runs} predictions)")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

# --- Save per-class metrics to JSON for thesis reuse ---
prec, rec, f1, sup = precision_recall_fscore_support(
    y_test, y_pred, labels=range(len(le.classes_)), zero_division=0
)
per_class = {
    cls: {
        "precision": float(prec[i]),
        "recall":    float(rec[i]),
        "f1":        float(f1[i]),
        "support":   int(sup[i]),
    }
    for i, cls in enumerate(le.classes_)
}
report_dict = classification_report(
    y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0
)
training_phase_keys = ["data_loading_total", "preprocessing", "feature_extraction",
                       "scaling_pca", "smoteenn", "gridsearch_rf", "final_training"]
total_training_time = sum(timings.get(k, 0) for k in training_phase_keys)

results = {
    "model": "RandomForest",
    "databases": ["MIT-BIH (44 records, paced excluded)", "MIT-BIH Supraventricular (78 records)"],
    "classes": list(le.classes_),
    "test_size": TEST_SIZE,
    "seed": SEED,
    "best_params": best_params,
    "accuracy": float(acc),
    "macro_f1": float(report_dict["macro avg"]["f1-score"]),
    "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
    "per_class": per_class,
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "n_train": int(len(y_train)),
    "n_train_balanced": int(len(y_train_bal)),
    "n_test": int(len(y_test)),
    "split_strategy": "patient-wise per database",
    "n_train_records": len(train_recs),
    "n_test_records":  len(test_recs),
    "train_records":   sorted(train_recs),
    "test_records":    sorted(test_recs),
    "timings_seconds": timings,
    "total_training_time_seconds": float(total_training_time),
    "total_training_time_hms": fmt_hms(total_training_time),
    "avg_prediction_latency_ms_batch": float(per_beat_ms),
    "single_beat_latency_ms": float(single_latency_ms),
}
with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f" Per-class metrics saved : {RESULTS_JSON}")

# ════════════════════════════════════════════════════════════
# STEP 9 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(" STEP 9 — FEATURE IMPORTANCE")
print("═" * 60)

# v6: RF operates in PCA space — feature names are components PC1…PCn.
# To map back to original features, inspect pca.components_ (shape: n_comp × 23).
n_components  = X_tr_pca.shape[1]
feature_names = [f"PC{i+1}" for i in range(n_components)]

importances = rf_final.feature_importances_
indices     = np.argsort(importances)[::-1]

print(" Top 10 most important features:")
for i in range(10):
    print(f"  {i+1:2d}. {feature_names[indices[i]]:<22} {importances[indices[i]]:.4f}")

# ════════════════════════════════════════════════════════════
# STEP 9b — PERMUTATION IMPORTANCE (robust feature importance)
# ════════════════════════════════════════════════════════════
from sklearn.inspection import permutation_importance
print("\n" + "═" * 60)
print(" STEP 9b — PERMUTATION IMPORTANCE (10 repeats)")
print("═" * 60)
with Timer("permutation_importance"):
    perm = permutation_importance(
        rf_final, X_te_pca, y_test,   # v6: use PCA test features
        n_repeats=10, random_state=SEED, n_jobs=-1,
    )
perm_indices = np.argsort(perm.importances_mean)[::-1]
print(" Top 10 features by permutation importance:")
for i in range(10):
    j = perm_indices[i]
    print(f"  {i+1:2d}. {feature_names[j]:<22} "
          f"{perm.importances_mean[j]:.4f} ± {perm.importances_std[j]:.4f}")

# Save permutation importance chart
plt.figure(figsize=(10, 5))
plt.bar(range(len(perm.importances_mean)),
        perm.importances_mean[perm_indices],
        yerr=perm.importances_std[perm_indices],
        color="darkorange", capsize=3)
plt.xticks(range(len(perm.importances_mean)),
           [feature_names[i] for i in perm_indices],
           rotation=45, ha="right", fontsize=8)
plt.title("Permutation Importance — Random Forest (3-class: N, S, V)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_perm_rf_3class_v6.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f" Permutation importance saved : "
      f"{OUTPUT_DIR}/feature_importance_perm_rf_3class_v6.png")

# Merge permutation importance into results JSON
results["permutation_importance"] = {
    feature_names[j]: {
        "mean": float(perm.importances_mean[j]),
        "std":  float(perm.importances_std[j]),
    } for j in perm_indices
}
with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f" Results JSON updated          : {RESULTS_JSON}")

# ════════════════════════════════════════════════════════════
# STEP 10 — SAVE PLOTS
# ════════════════════════════════════════════════════════════
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix — RF, AAMI 3-class: N, S, V (MIT-BIH + svdb)")
plt.savefig(f"{OUTPUT_DIR}/cm_rf_3class_v6.png", dpi=150, bbox_inches="tight")
plt.close()
print(f" Confusion matrix saved   : {OUTPUT_DIR}/cm_rf_3class_v6.png")

plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), importances[indices], color="steelblue")
plt.xticks(range(len(importances)),
           [feature_names[i] for i in indices], rotation=45, ha="right", fontsize=8)
plt.title("Feature Importances — Random Forest (3-class: N, S, V)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_rf_3class_v6.png", dpi=150, bbox_inches="tight")
plt.close()
print(f" Feature importance saved : {OUTPUT_DIR}/feature_importance_rf_3class_v6.png")

# ════════════════════════════════════════════════════════════
# TIMING SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60); print(" TIMING SUMMARY"); print("═" * 60)
for k, v in timings.items():
    if k.endswith("_ms"):
        print(f" {k:<35} {v:8.3f} ms")
    else:
        print(f" {k:<35} {v:8.2f} s   ({v/60:6.2f} min)")
print("-" * 60)
print(f" TOTAL TRAINING TIME (sum of stages above): {fmt_hms(total_training_time)}")
print(f" Average prediction latency per beat      : {per_beat_ms:.3f} ms")
print(f" Single-beat predict latency (real-time)  : {single_latency_ms:.3f} ms")

print("\n" + "═" * 60)
print(" ALL DONE — review outputs in:", OUTPUT_DIR)
print("═" * 60)
