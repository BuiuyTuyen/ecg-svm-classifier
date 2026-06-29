"""
Quick beat counter for the 3-class AAMI scheme — does NOT train any model.
Loads MIT-BIH (44 records, paced excluded) + svdb (78 records), maps to 3-class,
applies the patient-wise split, and prints counts for the thesis dataset chapter.

Runtime: ~3-5 minutes.
"""
import os
import json
from collections import Counter
import numpy as np
import wfdb
from scipy.signal import resample

FS_TARGET       = 360
HALF_WIN_TARGET = 90
WINDOW_TARGET   = 2 * HALF_WIN_TARGET

CLASSES = ["N", "S", "V"]
BEAT_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "S": "S", "A": "S", "a": "S", "J": "S",
    "V": "V", "E": "V", "F": "V",
}

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
OUTPUT_JSON = "./outputs/beat_counts_3class.json"


def count_record(rec, db_path, fs_native):
    path = os.path.join(db_path, rec)
    if not os.path.exists(path + ".hea"):
        return None
    ann = wfdb.rdann(path, "atr")
    half = int(round(0.25 * fs_native))
    sig_len = wfdb.rdrecord(path, sampto=1).sig_len  # length not loaded
    record = wfdb.rdrecord(path)
    sig_len = len(record.p_signal[:, 0])
    counts = Counter()
    for idx, sym in zip(ann.sample, ann.symbol):
        if sym not in BEAT_MAP:
            continue
        s, e = idx - half, idx + half
        if s < 0 or e > sig_len:
            continue
        counts[BEAT_MAP[sym]] += 1
    return dict(counts)


print("Counting beats — MIT-BIH...")
mitdb_per_record = {}
for rec in MITDB_RECORDS:
    counts = count_record(rec, LOCAL_MITDB_PATH, fs_native=360)
    if counts is None:
        print(f"  {rec}: missing")
        continue
    mitdb_per_record[f"mitdb_{rec}"] = counts
    print(f"  {rec}: {counts}")

print("\nCounting beats — svdb...")
svdb_per_record = {}
for rec in SVDB_RECORDS:
    counts = count_record(rec, LOCAL_SVDB_PATH, fs_native=128)
    if counts is None:
        print(f"  {rec}: missing")
        continue
    svdb_per_record[f"svdb_{rec}"] = counts
    print(f"  {rec}: {counts}")

# Aggregate
def agg(d):
    total = Counter()
    for v in d.values():
        total.update(v)
    return dict(total)

mitdb_total = agg(mitdb_per_record)
svdb_total  = agg(svdb_per_record)
grand_total = dict(Counter(mitdb_total) + Counter(svdb_total))

# Patient-wise 80/20 split: 80% of records per database go to train
import random
random.seed(42)

mitdb_keys = sorted(mitdb_per_record.keys())
svdb_keys  = sorted(svdb_per_record.keys())
n_train_mitdb = int(0.8 * len(mitdb_keys))
n_train_svdb  = int(0.8 * len(svdb_keys))
random.shuffle(mitdb_keys)
random.shuffle(svdb_keys)
train_keys = mitdb_keys[:n_train_mitdb] + svdb_keys[:n_train_svdb]
test_keys  = mitdb_keys[n_train_mitdb:] + svdb_keys[n_train_svdb:]

train_counts = Counter()
test_counts  = Counter()
for k in train_keys:
    train_counts.update((mitdb_per_record | svdb_per_record)[k])
for k in test_keys:
    test_counts.update((mitdb_per_record | svdb_per_record)[k])

summary = {
    "config": {
        "classes": CLASSES,
        "beat_map": BEAT_MAP,
        "mitdb_records_used": len(mitdb_per_record),
        "svdb_records_used":  len(svdb_per_record),
        "paced_excluded": sorted(PACED_RECORDS),
    },
    "totals": {
        "mitdb": mitdb_total,
        "svdb":  svdb_total,
        "combined": grand_total,
        "combined_grand_total": sum(grand_total.values()),
    },
    "patient_wise_split_80_20": {
        "train_records": len(train_keys),
        "test_records": len(test_keys),
        "train_class_counts": dict(train_counts),
        "test_class_counts": dict(test_counts),
        "train_total": sum(train_counts.values()),
        "test_total": sum(test_counts.values()),
    },
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("BEAT COUNT SUMMARY (3-class AAMI: N, S, V)")
print("=" * 60)
print(f"MIT-BIH (44 records, paced excluded): {mitdb_total} = {sum(mitdb_total.values())}")
print(f"svdb    (78 records):                  {svdb_total} = {sum(svdb_total.values())}")
print(f"COMBINED:                              {grand_total} = {sum(grand_total.values())}")
print()
print(f"Patient-wise 80/20 split:")
print(f"  Train: {len(train_keys)} records, {dict(train_counts)} = {sum(train_counts.values())} beats")
print(f"  Test : {len(test_keys)} records, {dict(test_counts)} = {sum(test_counts.values())} beats")
print(f"\nSaved to: {OUTPUT_JSON}")
