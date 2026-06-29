"""
predict.py — Module du doan nhip tim tu tin hieu ECG moi (v6)
=============================================================
Tuong thich voi mo hinh v6 (KNN / SVM / RF): bundle chua scaler, pca,
label_enc va classifier.

Vi du su dung:
    from predict import predict_record
    results = predict_record("path/to/record", "outputs/ecg_rf_3class_v6.pkl")
    for r in results:
        print(r['beat'], r['r_peak'], r['label'], r.get('confidence'))
"""
import numpy as np
import wfdb
import joblib
from scipy.signal import butter, filtfilt, find_peaks
from ecg_features import extract_features

FS       = 360
HALF_WIN = 90


def bandpass_filter(sig, fs=360, lo=0.5, hi=40, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, sig)


def detect_r_peaks(sig, fs=360):
    d   = np.diff(sig) ** 2
    w   = int(0.15 * fs)
    itg = np.convolve(d, np.ones(w) / w, mode='same')
    peaks, _ = find_peaks(itg, distance=int(0.3 * fs), height=np.mean(itg))
    return peaks


def predict_record(record_path, model_path='outputs/ecg_rf_3class_v6.pkl'):
    """
    Du doan nhan nhip tim cho toan bo tin hieu ECG trong mot record.

    Parameters
    ----------
    record_path : str
        Duong dan den record WFDB (khong co phan mo rong, vd. "mitdb/100").
    model_path  : str
        Duong dan den file .pkl chua bundle mo hinh v6.

    Returns
    -------
    list[dict]
        Moi phan tu la {'beat': int, 'r_peak': int, 'label': str,
        'confidence': float (neu mo hinh ho tro predict_proba)}.
    """
    bundle = joblib.load(model_path)
    scaler = bundle['scaler']
    pca    = bundle['pca']
    le     = bundle['label_enc']
    # Tim khoa classifier phu hop (knn / svm / rf)
    clf_key = next(k for k in ('knn', 'svm', 'rf') if k in bundle)
    clf     = bundle[clf_key]

    rec  = wfdb.rdrecord(record_path)
    sig  = bandpass_filter(rec.p_signal[:, 0])
    peaks = detect_r_peaks(sig)

    results = []
    for i, idx in enumerate(peaks):
        s, e = idx - HALF_WIN, idx + HALF_WIN
        if s < 0 or e > len(sig):
            continue
        # RR tinh bang giay (nhat quan voi pipeline huan luyen)
        pre_rr  = (idx - peaks[i - 1]) / FS if i > 0 else 1.0
        post_rr = (peaks[i + 1] - idx) / FS if i < len(peaks) - 1 else pre_rr

        feat     = np.array(extract_features(sig[s:e], pre_rr, post_rr)).reshape(1, -1)
        feat_sc  = scaler.transform(feat)
        feat_pca = pca.transform(feat_sc)

        pred  = clf.predict(feat_pca)[0]
        label = le.inverse_transform([pred])[0]

        entry = {'beat': i, 'r_peak': int(idx), 'label': label}
        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(feat_pca)[0]
            entry['confidence'] = float(max(proba))
        results.append(entry)

    return results
