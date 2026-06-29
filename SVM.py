"""
ECG SVM Classifier v6 — MIT-BIH + svdb, AAMI 3-class + timing instrumentation
=============================================================================
Changes from v5 (review corrections):
  - [FIX] GridSearch changed from 3-fold to 5-fold CV — consistent with
    KNN (was 3-fold, KNN was 5-fold; inconsistent experimental conditions)

Changes from v3 (v5):
  - 3-class AAMI EC57: N, S, V (F merged into V, Q excluded)
  - Added MIT-BIH Supraventricular Arrhythmia Database (svdb, 78 records)
    resampled from 128 Hz to 360 Hz to match MIT-BIH
  - Excluded paced records 102, 104, 107, 217 (Chazal et al. 2004)
  - Wall-clock timing for every pipeline stage saved to results JSON
  - Per-beat prediction latency reported (for thesis Chapter 4)
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
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
FS_TARGET       = 360
HALF_WIN_TARGET = 90
WINDOW_TARGET   = 2 * HALF_WIN_TARGET

CLASSES = ["N", "S", "V"]
BEAT_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "S": "S", "A": "S", "a": "S", "J": "S",
    "V": "V", "E": "V", "F": "V",   # F (fusion) merged into V per AAMI recommendation
    # Q and f excluded: unclassifiable beats not in AAMI EC57 evaluation classes
}

TEST_SIZE    = 0.20
SEED         = 42
OUTPUT_DIR   = "./outputs"
MODEL_PATH   = f"{OUTPUT_DIR}/ecg_svm_3class_v6.pkl"
RESULTS_JSON = f"{OUTPUT_DIR}/results_svm_3class_v6.json"

PACED_RECORDS = {"102", "104", "107", "217"}
MITDB_ALL = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
MITDB_RECORDS = [r for r in MITDB_ALL if r not in PACED_RECORDS]

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
# TIMING UTILITY
# ════════════════════════════════════════════════════════════
timings = {}

class Timer:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *args):
        dt = time.perf_counter() - self.t0
        timings[self.label] = dt
        print(f"  ⏱ {self.label}: {dt:.2f} s ({dt/60:.2f} min)")

def fmt_hms(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING (MIT-BIH + svdb)
# ════════════════════════════════════════════════════════════
print("═" * 60)
print(" STEP 1 — DATA LOADING (MIT-BIH + svdb, AAMI 3-class: N, S, V)")
print("═" * 60)

if not os.path.isdir(LOCAL_MITDB_PATH):
    raise FileNotFoundError(f"MIT-BIH folder not found: {LOCAL_MITDB_PATH}")


def ensure_svdb_local():
    hea_count = (sum(1 for f in os.listdir(LOCAL_SVDB_PATH) if f.endswith(".hea"))
                 if os.path.isdir(LOCAL_SVDB_PATH) else 0)
    if hea_count >= 70:
        print(f"  svdb found locally ({hea_count} .hea files)")
        return
    print(f"  svdb not found — downloading from PhysioNet (~50 MB, one-time)…")
    os.makedirs(LOCAL_SVDB_PATH, exist_ok=True)
    wfdb.dl_database("svdb", dl_dir=LOCAL_SVDB_PATH)
    print("  svdb download complete")


def load_record(rec, db_path, fs_native):
    record_path = os.path.join(db_path, rec)
    record = wfdb.rdrecord(record_path)
    ann    = wfdb.rdann(record_path, "atr")
    sig    = record.p_signal[:, 0]
    half_win_native = int(round(0.25 * fs_native))

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
        beat = resample(beat_native, WINDOW_TARGET) if fs_native != FS_TARGET else beat_native
        beats.append(beat)
        labels.append(BEAT_MAP[sym])

        if i > 0:
            pre_rr_sec = (beat_samples[i] - beat_samples[i-1]) / fs_native
        else:
            pre_rr_sec = (beat_samples[1] - beat_samples[0]) / fs_native if len(beat_samples) > 1 else 1.0
        post_rr_sec = (beat_samples[i+1] - beat_samples[i]) / fs_native if i < len(beat_samples)-1 else pre_rr_sec
        pre_rrs.append(pre_rr_sec)
        post_rrs.append(post_rr_sec)
    return beats, labels, pre_rrs, post_rrs


with Timer("data_loading_total"):
    with Timer("load_mitdb"):
        print(f"\n Loading MIT-BIH Arrhythmia ({len(MITDB_RECORDS)} records, paced excluded)…")
        mitdb_b, mitdb_l, mitdb_pre, mitdb_post, mitdb_recs = [], [], [], [], []
        for rec in MITDB_RECORDS:
            try:
                b, l, pre, post = load_record(rec, LOCAL_MITDB_PATH, fs_native=360)
                mitdb_b.extend(b); mitdb_l.extend(l); mitdb_pre.extend(pre); mitdb_post.extend(post)
                mitdb_recs.extend([f"mitdb_{rec}"] * len(b))
                print(f"  MIT-BIH {rec}: {len(b)} beats")
            except FileNotFoundError:
                print(f"  MIT-BIH {rec}: file not found — skipped")
        print(f"  MIT-BIH subtotal: {len(mitdb_b)} beats")

    ensure_svdb_local()
    with Timer("load_svdb"):
        print(f"\n Loading svdb ({len(SVDB_RECORDS)} records, 128→360 Hz)…")
        svdb_b, svdb_l, svdb_pre, svdb_post, svdb_recs = [], [], [], [], []
        for rec in SVDB_RECORDS:
            try:
                b, l, pre, post = load_record(rec, LOCAL_SVDB_PATH, fs_native=128)
                svdb_b.extend(b); svdb_l.extend(l); svdb_pre.extend(pre); svdb_post.extend(post)
                svdb_recs.extend([f"svdb_{rec}"] * len(b))
                print(f"  svdb {rec}: {len(b)} beats")
            except FileNotFoundError:
                print(f"  svdb {rec}: file not found — skipped")
        print(f"  svdb subtotal: {len(svdb_b)} beats")

all_b   = mitdb_b   + svdb_b
all_l   = mitdb_l   + svdb_l
all_pre = mitdb_pre + svdb_pre
all_post= mitdb_post+ svdb_post
all_recs= mitdb_recs+ svdb_recs
print(f"\n COMBINED TOTAL: {len(all_b)} beats")
print(f" Class distribution: {dict(sorted(Counter(all_l).items()))}")

# ════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESSING (IQR outlier removal)
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60); print(" STEP 2 — PREPROCESSING"); print("═" * 60)

with Timer("preprocessing"):
    X_all   = np.array(all_b, dtype=float)
    y_all   = np.array(all_l)
    pre_all = np.array(all_pre, dtype=float)
    post_all= np.array(all_post, dtype=float)
    recs_all= np.array(all_recs)
    amp     = X_all.max(axis=1) - X_all.min(axis=1)   # kept for post-split outlier removal
    print(f" Beats loaded: {len(y_all)}")
    print(f" Class distribution: {dict(sorted(Counter(y_all).items()))}")

# ════════════════════════════════════════════════════════════
# STEP 3 — FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60); print(" STEP 3 — FEATURE EXTRACTION"); print("═" * 60)


with Timer("feature_extraction"):
    X_feat = np.nan_to_num(np.array([
        extract_features(b, pr, po)
        for b, pr, po in zip(X_all, pre_all, post_all)
    ]))
    print(f" Feature matrix: {X_feat.shape}")

# ════════════════════════════════════════════════════════════
# STEP 4 — 80/20 SPLIT
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60); print(" STEP 4 — PATIENT-WISE 80/20 SPLIT (per database)"); print("═" * 60)
le    = LabelEncoder()
y_enc = le.fit_transform(y_all)

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
# STEP 5 — SCALING + PCA
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60); print(" STEP 5 — SCALING + PCA"); print("═" * 60)
with Timer("scaling_pca"):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    pca = PCA(n_components=0.95, random_state=SEED)
    X_tr_pca = pca.fit_transform(X_tr_sc)
    X_te_pca = pca.transform(X_te_sc)
    print(f" PCA: {X_train.shape[1]} → {X_tr_pca.shape[1]} components "
          f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

# ════════════════════════════════════════════════════════════
# STEP 6 — SMOTEENN
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60); print(" STEP 6 — SMOTEENN BALANCING"); print("═" * 60)
print(" This step may take 5–15 minutes on a large combined dataset…")
with Timer("smoteenn"):
    smoteenn = SMOTEENN(random_state=SEED)
    X_tr_bal, y_tr_bal = smoteenn.fit_resample(X_tr_pca, y_train)
    print(f" After balancing: {dict(zip(le.classes_, np.bincount(y_tr_bal)))}")

# Cap per-class samples at 50k to keep SVM tractable
MAX_PER_CLASS = 50_000
rng = np.random.default_rng(SEED)
keep_idx = []
for cls in np.unique(y_tr_bal):
    idx = np.where(y_tr_bal == cls)[0]
    if len(idx) > MAX_PER_CLASS:
        idx = rng.choice(idx, MAX_PER_CLASS, replace=False)
    keep_idx.append(idx)
keep_idx = np.concatenate(keep_idx)
X_tr_bal, y_tr_bal = X_tr_bal[keep_idx], y_tr_bal[keep_idx]
print(f" After capping (≤{MAX_PER_CLASS}/class): {dict(zip(le.classes_, np.bincount(y_tr_bal)))}")

# ════════════════════════════════════════════════════════════
# STEP 7 — SVM GRIDSEARCH
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60); print(" STEP 7 — SVM GRIDSEARCH (5-fold CV)"); print("═" * 60)
# Changed from 3-fold to 5-fold to match KNN — consistent experimental conditions.
param_grid = {"C": [0.1, 1, 10], "gamma": ["scale"], "kernel": ["rbf"]}
print(f" Grid: 3 × 1 = 3 combinations × 5-fold = 15 fits")
with Timer("gridsearch_svm"):
    svm_base = SVC(class_weight="balanced", probability=True, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid_search = GridSearchCV(svm_base, param_grid, cv=cv,
                               scoring="f1_macro", n_jobs=-1, verbose=2)
    grid_search.fit(X_tr_bal, y_tr_bal)
    best_params = grid_search.best_params_
    print(f"\n Best params      : {best_params}")
    print(f" Best CV f1_macro : {grid_search.best_score_:.4f}")

# ════════════════════════════════════════════════════════════
# STEP 8 — FINAL MODEL
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60); print(" STEP 8 — FINAL MODEL TRAINING"); print("═" * 60)
with Timer("final_training"):
    svm_final = SVC(C=best_params["C"], gamma=best_params["gamma"],
                    kernel="rbf", class_weight="balanced",
                    probability=True, random_state=SEED)
    svm_final.fit(X_tr_bal, y_tr_bal)

# ════════════════════════════════════════════════════════════
# STEP 9 — PREDICTION + LATENCY BENCHMARK
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60); print(" STEP 9 — TEST EVALUATION + LATENCY"); print("═" * 60)
with Timer("prediction_total"):
    y_pred = svm_final.predict(X_te_pca)
n_test = len(y_test)
per_beat_ms = timings["prediction_total"] / n_test * 1000.0
print(f"  • Total predict time: {timings['prediction_total']:.3f} s for {n_test} beats")
print(f"  • Avg latency / beat: {per_beat_ms:.3f} ms")

# Single-beat latency benchmark (more realistic for real-time inference)
single_beat = X_te_pca[:1]
n_single_runs = 1000
t0 = time.perf_counter()
for _ in range(n_single_runs):
    _ = svm_final.predict(single_beat)
single_latency_ms = (time.perf_counter() - t0) / n_single_runs * 1000.0
timings["single_beat_latency_ms"] = single_latency_ms
print(f"  • Single-beat latency: {single_latency_ms:.3f} ms (avg of {n_single_runs} predictions)")

acc = accuracy_score(y_test, y_pred)
print(f"\n Accuracy : {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

# ════════════════════════════════════════════════════════════
# STEP 10 — SAVE EVERYTHING
# ════════════════════════════════════════════════════════════
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump({
    "scaler": scaler, "pca": pca, "svm": svm_final, "label_enc": le,
    "best_params": best_params, "window": WINDOW_TARGET, "fs": FS_TARGET,
    "classes": list(le.classes_), "test_size": TEST_SIZE, "has_rr": True,
    "schema": "AAMI_3class_v6",
}, MODEL_PATH)
print(f"\n Model saved: {MODEL_PATH}")

prec, rec, f1, sup = precision_recall_fscore_support(
    y_test, y_pred, labels=range(len(le.classes_)), zero_division=0)
per_class = {cls: {"precision": float(prec[i]), "recall": float(rec[i]),
                   "f1": float(f1[i]), "support": int(sup[i])}
             for i, cls in enumerate(le.classes_)}
report_dict = classification_report(y_test, y_pred, target_names=le.classes_,
                                    output_dict=True, zero_division=0)

# Sum training-related times (everything except prediction)
training_phase_keys = ["data_loading_total", "preprocessing", "feature_extraction",
                       "scaling_pca", "smoteenn", "gridsearch_svm", "final_training"]
total_training_time = sum(timings.get(k, 0) for k in training_phase_keys)

results = {
    "model": "SVM",
    "databases": ["MIT-BIH (44 records, paced excluded)", "MIT-BIH Supraventricular (78 records)"],
    "classes": list(le.classes_),
    "test_size": TEST_SIZE, "seed": SEED,
    "best_params": best_params,
    "accuracy": float(acc),
    "macro_f1": float(report_dict["macro avg"]["f1-score"]),
    "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
    "per_class": per_class,
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "n_train": int(len(y_train)),
    "n_train_balanced": int(len(y_tr_bal)),
    "n_test": int(n_test),
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
print(f" Results JSON saved: {RESULTS_JSON}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix — SVM, AAMI 3-class: N, S, V (MIT-BIH + svdb)")
plt.savefig(f"{OUTPUT_DIR}/cm_svm_3class_v6.png", dpi=150, bbox_inches="tight")
plt.close()
print(f" Confusion matrix : {OUTPUT_DIR}/cm_svm_3class_v6.png")

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
