"python src/train_iforest.py  --features outputs/split_user_day/features_train.csv  --model-dir outputs/models_if_userday  --scores-out outputs/scores/scores_iforest_train.csv  --contamination 0.01  --n-estimators 1500  --max-samples 4096 -v"
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Các cột định danh nên loại khỏi X
ID_COLS_DEFAULT = ["user","date","session_id","pc","department","role","is_admin","is_contractor"]

def _read_any(p: str) -> pd.DataFrame:
    pth = Path(p)
    if not pth.exists():
        raise FileNotFoundError(p)
    if pth.suffix.lower() == ".parquet":
        return pd.read_parquet(pth)
    return pd.read_csv(pth)

def _write_any(df: pd.DataFrame, p: Path, fmt: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)

def _pick_numeric(df: pd.DataFrame, drop_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Giữ lại cột số để đưa vào IF."""
    to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=to_drop, errors="ignore")
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    return X[num_cols], num_cols

def _percentiles(s: pd.Series, ps=(0.95,0.99,0.995)) -> Dict[str, float]:
    q = s.quantile(list(ps))
    return {f"p{int(p*1000)/10:g}": float(q.iloc[i]) for i, p in enumerate(ps)}

def main():
    ap = argparse.ArgumentParser("Train Isolation Forest")
    ap.add_argument("--features", required=True, help="CSV/Parquet features.")
    ap.add_argument("--model-dir", default="outputs/models_iforest")
    ap.add_argument("--scores-out", default=None, help="Xuất điểm train.")
    ap.add_argument("--out-format", choices=["csv","parquet"], default="csv")

    # IF params
    ap.add_argument("--contamination", type=float, default=0.01)
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--max-samples", default=1024,
                    help="int hoặc float (tỷ lệ). Ví dụ 0.8 hoặc 2048.")
    ap.add_argument("--random-state", type=int, default=42)

    # Preprocess params
    ap.add_argument("--id-cols", default=",".join(ID_COLS_DEFAULT),
                    help="các cột loại khỏi X, ngăn cách bằng dấu phẩy")
    ap.add_argument("--cap-p", type=float, default=0.995,
                    help="percentile để clip outlier, vd 0.995 = p99.5")
    ap.add_argument("--auto-log1p", action="store_true", default=False,
                    help="tự động log1p các cột có skewness lớn")
    ap.add_argument("--log1p-cols", default="",
                    help="chỉ định danh sách cột log1p, ví dụ: cnt_http,size_sum")
    ap.add_argument("--log1p-skew-th", type=float, default=1.0,
                    help="ngưỡng |skew| để auto-log1p khi --auto-log1p")

    ap.add_argument("-v","--verbose", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ---- Đọc & chuẩn hoá cột date (nếu có) ----
    df = _read_any(args.features)
    if "date" in df.columns:
        # Chuẩn hoá về DATE (ngày) để xuất QC cho dễ đọc; không ảnh hưởng X
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # ---- Chọn cột số để train ----
    drop_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    X0, used_cols = _pick_numeric(df, drop_cols)
    X = X0.copy()

    # ---- Impute đơn giản: NaN -> 0 ----
    X = X.fillna(0.0)

    # ---- Clip theo p (tính trên TRAIN) ----
    cap_p = float(args.cap_p)
    caps: Dict[str, float] = {}
    for c in used_cols:
        try:
            cap = float(np.nanquantile(X[c].values, cap_p))
        except Exception:
            cap = None
        if cap is not None and np.isfinite(cap):
            caps[c] = cap
            X[c] = np.minimum(X[c], cap)

    # ---- Chọn cột log1p ----
    log1p_cols: List[str] = []
    if args.log1p_cols.strip():
        # user chỉ định
        log1p_cols = [c.strip() for c in args.log1p_cols.split(",") if c.strip() and c in X.columns]
    elif args.auto_log1p:
        # tự động theo skewness
        skew = X.skew(numeric_only=True)
        log1p_cols = [c for c, v in skew.items() if np.isfinite(v) and abs(v) >= args.log1p_skew_th]
    # áp dụng log1p (an toàn cho số 0)
    for c in log1p_cols:
        X[c] = np.log1p(np.clip(X[c], a_min=0, a_max=None))

    # ---- Scale (fit trên TRAIN) ----
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if args.verbose:
        print(f"[train] rows={len(X):,} | features={len(used_cols)}")
        # In một ít QC cho 8 cột đầu
        sample_cols = used_cols[:8]
        qc = {c: _percentiles(X0[c]) for c in sample_cols if c in X0}
        print("[train] percentiles (before clip) for sample cols:", json.dumps(qc, indent=2))

    # ---- Fit IF ----
    # max_samples: cho phép int/float
    max_samples = args.max_samples
    if isinstance(max_samples, str):
        max_samples = float(max_samples) if "." in max_samples else int(max_samples)

    clf = IsolationForest(
        n_estimators=int(args.n_estimators),
        max_samples=max_samples,
        contamination=float(args.contamination),
        random_state=int(args.random_state),
        n_jobs=-1,
    )
    clf.fit(Xs)

    # ---- Lưu model & artefacts ----
    joblib.dump(clf, model_dir / "iforest_model.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")

    meta = {
        "used_cols": used_cols,
        "drop_cols": drop_cols,
        "params": {
            "contamination": args.contamination,
            "n_estimators": args.n_estimators,
            "max_samples": args.max_samples,
            "random_state": args.random_state,
            "cap_p": args.cap_p,
            "auto_log1p": bool(args.auto_log1p),
            "log1p_cols_cli": args.log1p_cols,
            "log1p_skew_th": args.log1p_skew_th,
        },
        "caps": {k: float(v) for k, v in caps.items()},
        "log1p_cols": log1p_cols,
        "n_rows_train": int(len(df)),
        "n_features_train": int(len(used_cols)),
    }
    (model_dir / "iforest_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (model_dir / "caps.json").write_text(json.dumps(meta["caps"], indent=2), encoding="utf-8")
    (model_dir / "log1p_cols.json").write_text(json.dumps(log1p_cols, indent=2), encoding="utf-8")

    # ---- Xuất điểm trên tập train ----
    if args.scores_out:
        # Lưu ý: decision_function cho giá trị lớn hơn khi bình thường → đảo dấu để
        # score_iforest lớn hơn = bất thường hơn (thống nhất với downstream).
        scores = -clf.decision_function(Xs)

        # cố gắng kèm các ID nếu có
        id_out = {}
        for c in ["user","date","session_id","pc","department","role","is_admin","is_contractor"]:
            if c in df.columns:
                id_out[c] = df[c]
        out = pd.DataFrame({**id_out, "score_iforest": scores})

        _write_any(out, Path(args.scores_out), args.out_format)
        if args.verbose:
            print(f"[train] wrote train scores -> {args.scores_out} ({args.out_format})")

if __name__ == "__main__":
    main()
