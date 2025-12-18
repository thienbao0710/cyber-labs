"python src\eval_iforest.py --scores outputs\scores\scores_iforest_test.csv --labels label\labels_user_day_full.csv --user-col user --date-col date --label-col label --Ks 10,20,50,100 --topN 200 --per-day --out-dir outputs\eval_if_userday"
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def _read_any(p):
    p = Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_csv(p)

def _to_date_col(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df

def _precision_at_k(y_true, y_score, K):
    if K <= 0: return float("nan")
    idx = np.argsort(-y_score)[:K]
    return float((y_true[idx]==1).sum() / max(K,1))

def _recall_at_k(y_true, y_score, K):
    pos = int((y_true==1).sum())
    if pos == 0 or K <= 0: return float("nan")
    idx = np.argsort(-y_score)[:K]
    tp = int((y_true[idx]==1).sum())
    return float(tp / pos)

def _fpr_at_k(y_true, y_score, K):
    # FPR = FP / TN trong top-K
    tn = int((y_true==0).sum())
    if tn == 0 or K <= 0: return float("nan")
    idx = np.argsort(-y_score)[:K]
    fp = int((y_true[idx]==0).sum())
    return float(fp / tn)

def _threshold_at_k(y_score, K):
    if K <= 0 or len(y_score)==0: return float("nan")
    K = min(K, len(y_score))
    # ngưỡng = điểm của phần tử thứ K sau khi sắp giảm dần
    s = np.sort(y_score)[::-1]
    return float(s[K-1])

def main():
    ap = argparse.ArgumentParser("Evaluate IF scores with labels (user-day friendly)")
    ap.add_argument("--scores", required=True, help="scores file (CSV/Parquet)")
    ap.add_argument("--labels", required=True, help="labels file (CSV/Parquet)")
    ap.add_argument("--user-col", default="user")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--score-col", default="", help="force using this column if provided")
    ap.add_argument("--K", type=int, default=50, help="legacy single-K (kept for backward-compat)")
    ap.add_argument("--Ks", default="", help="comma list, e.g. 10,20,50,100")
    ap.add_argument("--topN", type=int, default=200, help="rows to export in topN files")
    ap.add_argument("--per-day", action="store_true", help="evaluate P@K, R@K averaged by date")
    ap.add_argument("--out-dir", default="outputs/eval_iforest")
    args = ap.parse_args()

    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    sc = _read_any(args.scores)
    lb = _read_any(args.labels)

    # Chuẩn hoá thời gian
    sc = _to_date_col(sc, args.date_col)
    lb = _to_date_col(lb, args.date_col)

    # Lựa chọn score_col
    cand = [c for c in ["score_rerank","final_score","score_iforest"] if c in sc.columns]
    score_col = args.score_col.strip() if args.score_col else (cand[0] if cand else None)
    if not score_col:
        raise KeyError("Không tìm thấy score column. Hãy truyền --score-col hoặc có một trong [score_rerank, final_score, score_iforest].")

    # Merge theo key
    key_cols = [k for k in [args.user_col, args.date_col] if k in sc.columns and k in lb.columns]
    if not key_cols:
        raise KeyError("Không có cột khoá chung giữa scores và labels. Cần ít nhất một trong [user, date].")
    sc = sc.drop_duplicates(subset=[*key_cols]).copy()
    lb = lb.drop_duplicates(subset=[*key_cols]).copy()

    data = sc.merge(lb[[*key_cols, args.label_col]], on=key_cols, how="left")
    data[args.label_col] = data[args.label_col].fillna(0).astype(int)

    # Chuẩn hoá vector
    y = data[args.label_col].astype(int).values
    s = pd.to_numeric(data[score_col], errors="coerce").fillna(-np.inf).values
    # Loại nan/inf (nếu có)
    ok = np.isfinite(s)
    if ok.sum() < len(s):
        data = data.loc[ok].copy()
        y = data[args.label_col].values
        s = data[score_col].values

    # Metrics tổng thể
    uniq = np.unique(y)
    roc = roc_auc_score(y, s) if len(uniq) > 1 else float("nan")
    ap = average_precision_score(y, s) if len(uniq) > 1 else float("nan")

    # Danh sách K để đánh giá
    Ks = []
    if args.K: Ks.append(int(args.K))
    if args.Ks.strip():
        Ks += [int(x.strip()) for x in args.Ks.split(",") if x.strip().isdigit()]
    Ks = sorted(set([k for k in Ks if k > 0]))

    rows = len(data)
    pos = int((y==1).sum())
    base_rate = (pos / rows) if rows else float("nan")

    metrics_rows = []
    for K in Ks:
        pk  = _precision_at_k(y, s, K)
        rk  = _recall_at_k(y, s, K)
        fpr = _fpr_at_k(y, s, K)
        thr = _threshold_at_k(s, K)
        lift = (pk / base_rate) if (base_rate and base_rate > 0) else float("nan")
        metrics_rows.append({
            "K": K,
            "Precision@K": pk,
            "Recall@K": rk,
            "FPR@K": fpr,
            "Lift@K": lift,
            "Threshold_at_K": thr
        })
    pd.DataFrame(metrics_rows).to_csv(outd/"metrics_by_K.csv", index=False)

    # Per-day (optional): trung bình P@K, R@K theo ngày
    daily_summary = None
    if args.per_day:
        # group theo ngày, tính P@K/R@K từng ngày (nếu ngày đó có dữ liệu)
        per_rows = []
        for d, g in data.groupby(args.date_col):
            yy = g[args.label_col].values
            ss = pd.to_numeric(g[score_col], errors="coerce").fillna(-np.inf).values
            for K in Ks:
                pk = _precision_at_k(yy, ss, min(K, len(yy)))
                rk = _recall_at_k(yy, ss, min(K, len(yy)))
                per_rows.append({"date": d, "K": K, "P@K": pk, "R@K": rk, "n": len(yy), "pos": int((yy==1).sum())})
        daily_df = pd.DataFrame(per_rows)
        if not daily_df.empty:
            daily_df.to_csv(outd/"daily_metrics.csv", index=False)
            daily_summary = (daily_df.groupby("K")[["P@K","R@K"]].mean().reset_index()
                             .rename(columns={"P@K":"P@K_mean_by_day","R@K":"R@K_mean_by_day"}))

    # Tạo topN & top200 (giữ tương thích cũ)
    score_sorted = data.sort_values(score_col, ascending=False).reset_index(drop=True)
    topN = max(int(args.topN), 1)
    score_sorted.head(topN).to_csv(outd/"topK.csv", index=False)
    score_sorted.head(200).to_csv(outd/"top200.csv", index=False)

    # Summary JSON
    summary = {
        "n_rows_eval": int(rows),
        "n_pos_eval": int(pos),
        "score_col": score_col,
        "roc_auc": float(roc),
        "pr_auc": float(ap),
        "base_rate": float(base_rate),
        "metrics_by_K_csv": str(outd/"metrics_by_K.csv"),
        "topK_csv": str(outd/"topK.csv"),
        "top200_csv": str(outd/"top200.csv")
    }
    if daily_summary is not None:
        summary["per_day_avg"] = daily_summary.to_dict(orient="records")

    (outd/"eval_iforest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Lưu: {outd/'eval_iforest_summary.json'}")
    print(f"ROC-AUC={summary['roc_auc']:.4f} | AUPRC={summary['pr_auc']:.4f} | base_rate={summary['base_rate']:.6f}")
    if Ks:
        print(" | ".join([f"K={r['K']}: P={r['Precision@K']:.3f} R={r['Recall@K']:.3f} FPR={r['FPR@K']:.5f} Lift={r['Lift@K']:.1f}"
                          for r in metrics_rows]))

if __name__ == "__main__":
    main()
