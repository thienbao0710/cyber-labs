"python src\score_iforest.py --model outputs\models_if_userday\iforest_model.joblib --features outputs\split_user_day\features_test.csv --out data\outputs\scores\scores_iforest_test.csv -v "
import argparse, json
from pathlib import Path
import pandas as pd, numpy as np
import joblib

ID_COLS_DEFAULT = ["user","date","session_id","pc","department","role","is_admin","is_contractor"]

def _read_any(p: str) -> pd.DataFrame:
    pth = Path(p)
    if not pth.exists(): raise FileNotFoundError(p)
    return pd.read_parquet(pth) if pth.suffix.lower()==".parquet" else pd.read_csv(pth)

def _write_any(df: pd.DataFrame, p: Path, fmt: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    return df.to_parquet(p, index=False) if fmt=="parquet" else df.to_csv(p, index=False)

def main():
    ap = argparse.ArgumentParser("Score features with a trained IF model (reproducible)")
    ap.add_argument("--model", required=True, help="path to iforest_model.joblib")
    ap.add_argument("--features", required=True, help="CSV/Parquet features (TEST)")
    ap.add_argument("--out", required=True, help="scores output file")
    ap.add_argument("--out-format", choices=["csv","parquet"], default="csv")
    ap.add_argument("--id-cols", default=",".join(ID_COLS_DEFAULT))
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.6)  # IF weight
    ap.add_argument("--beta",  type=float, default=0.4)  # aux weight
    ap.add_argument("-v","--verbose", action="store_true")
    args = ap.parse_args()

    df = _read_any(args.features)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    model_path = Path(args.model)
    model_dir  = model_path.parent

    # ---- Load artefacts giống lúc train ----
    clf    = joblib.load(model_path)
    scaler = joblib.load(model_dir / "scaler.joblib")

    # meta: used_cols, caps, log1p_cols
    meta_path = model_dir / "iforest_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    used_cols   = meta.get("used_cols", [])
    caps        = meta.get("caps", {})
    log1p_cols  = meta.get("log1p_cols", [])

    # ---- Build X: chỉ lấy đúng used_cols, fill missing, reorder ----
    # loại ID trước khi lấy used_cols để tránh trùng
    drop_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    df_num = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    # thêm cột thiếu với 0
    for c in used_cols:
        if c not in df_num.columns:
            df_num[c] = 0.0
    # bỏ cột thừa
    X = df_num[used_cols].copy().fillna(0.0)

    # ---- Áp clip (caps) theo train ----
    for c, cap in caps.items():
        if c in X.columns and np.isfinite(cap):
            X[c] = np.minimum(X[c].astype(float), float(cap))

    # ---- Áp log1p cho đúng các cột đã dùng ở train ----
    for c in log1p_cols:
        if c in X.columns:
            X[c] = np.log1p(np.clip(X[c].astype(float), a_min=0, a_max=None))

    # ---- Scale và chấm điểm ----
    Xs = scaler.transform(X)
    score_if = -clf.decision_function(Xs)  # đảo dấu: lớn = bất thường hơn

    # ---- Gói output ----
    id_out = {}
    for c in ["user","date","session_id","pc","department","role","is_admin","is_contractor"]:
        if c in df.columns: id_out[c] = df[c]
    out = pd.DataFrame({**id_out, "score_iforest": score_if})

    # Rerank nếu có cột phụ (severity/…)
    if args.rerank:
        sev_col = None
        for cand in ["severity","sev","risk","rank_day","rank_global"]:
            if cand in df.columns: sev_col = cand; break
        if sev_col is not None:
            sev = (df[sev_col] - df[sev_col].min()) / (df[sev_col].max() - df[sev_col].min() + 1e-9)
            out["score_rerank"] = args.alpha*out["score_iforest"] + args.beta*sev.values
        else:
            out["score_rerank"] = out["score_iforest"]

    _write_any(out, Path(args.out), args.out_format)
    if args.verbose:
        print(f"[score] wrote -> {args.out} rows={len(out)} "
              f"| used_cols={len(used_cols)} | log1p={len(log1p_cols)}")

if __name__ == "__main__":
    main()
