"Bản features_full chứa toàn bộ features từ nguồn logs chính, dùng để phân tích"
"python src/make_features.py  --logs data/clear_logs/logs_std.parquet  --out artifacts/features_user_day.csv  --ldap-dir data/cert/LDAP  --catalog-out artifacts/ldap_catalog.json  -v"
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from datetime import datetime

pd.options.mode.copy_on_write = True


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------
# Chuẩn hoá tối thiểu các cột
# ------------------------------------------------------------

def _ensure_min_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đảm bảo DataFrame có đủ các cột tối thiểu với tên chuẩn:
        timestamp, user, pc, activity, url, filename, success, action
    Nếu thiếu sẽ tạo cột với NaN, hoặc rename từ 'table' -> 'activity', 'op' -> 'action'.
    """
    # activity: nếu chưa có thì lấy từ table
    if "activity" not in df.columns:
        if "table" in df.columns:
            df = df.rename(columns={"table": "activity"})
        else:
            df["activity"] = np.nan

    # Nếu không có action mà có op (từ logs_std) thì dùng op làm action
    if "action" not in df.columns and "op" in df.columns:
        df["action"] = df["op"]

    # Bổ sung các cột tối thiểu
    for col in ["timestamp", "user", "pc", "activity", "url",
                "filename", "success", "action"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


def _prepare_cols(df: pd.DataFrame, work_start: int, work_end: int) -> pd.DataFrame:
    """
    Chuẩn hoá timestamp/user, tạo các cờ logic:
        is_logon, is_http, is_file, is_device,
        is_failed_logon, is_file_delete,
        is_after_hours
    và cột date (ngày LOCAL, không chơi UTC nữa).
    """
    # 1) Parse timestamp từ logs_std -> để pandas tự nhận, KHÔNG ép utc=True
    ts = pd.to_datetime(df["timestamp"], errors="coerce")

    # Nếu có timezone (ví dụ +07:00) thì bỏ timezone, giữ nguyên giờ local
    if getattr(ts.dtype, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)

    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"])

    # 2) Lọc & chuẩn hoá user
    df = df[df["user"].notna() & (df["user"] != "")]
    df["user"] = df["user"].astype(str).str.strip().str.upper()

    # 3) Giờ & ngày (local)
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.normalize().astype("datetime64[ns]")

    # 4) Loại log theo activity
    act = df["activity"].astype("string").str.lower()

    # "logonfailed" cũng được tính là logon (nhưng sẽ là failed)
    df["is_logon"] = act.str.startswith("logon")   # match: logon, logonfailed, ...

    df["is_http"] = act.eq("http")
    df["is_file"] = act.eq("file")
    df["is_device"] = act.eq("device")

    # 5) failed_logon: dùng cả cột success và activity
    if "success" in df.columns:
        succ = df["success"].astype(str).str.lower()

        fail_by_success = succ.isin(["false", "0", "fail", "failed"])
        fail_by_activity = act.eq("logonfailed")

        df["is_failed_logon"] = df["is_logon"] & (fail_by_success | fail_by_activity)
    else:
        # fallback: nếu không có cột success thì coi logonfailed là fail
        df["is_failed_logon"] = act.eq("logonfailed")

    # 6) file_delete giữ nguyên logic cũ
    delete_pattern = r"delete|removed?|remove|rm|del"

    if "action" in df.columns or "op" in df.columns:
        src = df["action"] if "action" in df.columns else df["op"]
        act_col = src.astype(str).str.lower()
        df["is_file_delete"] = df["is_file"] & act_col.str.contains(
            delete_pattern, regex=True, na=False
        )
    else:
        df["is_file_delete"] = df["is_file"] & df["filename"].astype(str).str.lower().str.contains(
            r"delete|removed?|trash|cleanup",
            case=False,
            regex=True,
            na=False,
        )

    # 7) Cờ giờ làm việc / ngoài giờ
    df["is_work_hour"] = df["hour"].between(work_start, work_end, inclusive="both")
    df["is_after_hours"] = ~df["is_work_hour"]

    return df


# ------------------------------------------------------------
# Aggregate user×day
# ------------------------------------------------------------

def _agg_user_day(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["user", "date"], dropna=False)

    # Tạo cột chỉ chứa filename của device
    df["device_filename_only"] = np.where(df["is_device"], df["filename"], np.nan)
    g = df.groupby(["user", "date"], dropna=False)

    out = pd.DataFrame({
        "user": g["user"].first(),
        "date": g["date"].first(),
        "events": g.size(),
        "logon_count": g["is_logon"].sum(),
        "failed_logon": g["is_failed_logon"].sum(),
        "http_count": g["is_http"].sum(),
        "file_ops": g["is_file"].sum(),
        "file_delete_count": g["is_file_delete"].sum(),
        "after_hours_events": g["is_after_hours"].sum(),
        "unique_pc": g["pc"].nunique(),
        "unique_url": g["url"].nunique(),
        "unique_filename": g["filename"].nunique(),
        "device_ops": g["is_device"].sum(),
        "device_filename_count": g["device_filename_only"].nunique(),
    }).reset_index(drop=True)

    return out


def _finalize_basic(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Bổ sung các feature dẫn xuất (after_hours_ratio, flags),
    đảm bảo đủ cột kể cả khi DataFrame rỗng.
    """
    if len(feat) == 0:
        base_cols = [
            "events",
            "logon_count",
            "failed_logon",
            "http_count",
            "file_ops",
            "file_delete_count",
            "after_hours_events",
            "unique_pc",
            "unique_url",
            "unique_filename",
            "device_ops",
            "device_filename_count",
        ]
        for c in base_cols:
            feat[c] = 0

    # Tỉ lệ hoạt động ngoài giờ
    feat["after_hours_ratio"] = np.where(
        feat["events"] > 0,
        feat["after_hours_events"] / feat["events"],
        0.0,
    )

    # Cờ hỗ trợ debug / phân tích: heavy_http, multi_pc
    med_http = float(feat["http_count"].median()) if len(feat) else 0.0
    feat["heavy_http_flag"] = (feat["http_count"] > med_http).astype(int)
    feat["multi_pc_flag"] = (feat["unique_pc"] >= 2).astype(int)

    return (
        feat.fillna(0)
        .sort_values(["date", "user"])
        .reset_index(drop=True)
    )


# ------------------------------------------------------------
# Batch reader (PyArrow streaming)
# ------------------------------------------------------------

def process_with_pyarrow(
    parquet_path: str,
    work_start: int,
    work_end: int,
    batch_rows: int,
    progress_every: int,
    verbose: bool,
) -> pd.DataFrame:
    """
    Đọc logs_std.parquet theo batch để tránh hết RAM, trả về feature user×day (chưa có z-score).
    """
    import pyarrow.dataset as ds

    scanner = ds.dataset(parquet_path, format="parquet").scanner(batch_size=batch_rows)
    feats, n_rows, n_batch = [], 0, 0

    for batch in scanner.to_batches():
        n_batch += 1
        pdf = batch.to_pandas()
        n_rows += len(pdf)

        pdf = _ensure_min_cols(pdf)
        pdf = _prepare_cols(pdf, work_start, work_end)
        feats.append(_agg_user_day(pdf))

        if verbose and (n_batch % max(1, progress_every) == 0):
            print(f"[{ts()}] [BATCH {n_batch:>4}] rows_in={n_rows:,}")

    if not feats:
        return pd.DataFrame(columns=["user", "date"])

    big = pd.concat(feats, ignore_index=True)

    # Gộp lại theo user×day lần cuối
    g2 = big.groupby(["user", "date"], dropna=False).agg(
        events=("events", "sum"),
        logon_count=("logon_count", "sum"),
        failed_logon=("failed_logon", "sum"),
        http_count=("http_count", "sum"),
        file_ops=("file_ops", "sum"),
        file_delete_count=("file_delete_count", "sum"),
        after_hours_events=("after_hours_events", "sum"),
        unique_pc=("unique_pc", "max"),
        unique_url=("unique_url", "max"),
        unique_filename=("unique_filename", "max"),
        device_ops=("device_ops", "sum"),
        device_filename_count=("device_filename_count", "max"),
    ).reset_index()

    return g2


def process_with_pandas(
    parquet_path: str,
    work_start: int,
    work_end: int,
    verbose: bool,
) -> pd.DataFrame:
    """
    Fallback nếu không dùng được PyArrow. Đọc toàn bộ parquet bằng pandas.
    """
    if verbose:
        print("[WARN] PyArrow không khả dụng, dùng pandas.read_parquet (có thể tốn RAM).")
    df = pd.read_parquet(parquet_path)
    df = _ensure_min_cols(df)
    df = _prepare_cols(df, work_start, work_end)
    return _agg_user_day(df)


# ------------------------------------------------------------
# LDAP (optional, chỉ để thêm thông tin context)
# ------------------------------------------------------------

LDAP_COLMAP = {
    "user": ["user", "user_id", "userid", "employee", "employee_id", "username", "name"],
    "role": ["role", "job_role", "title", "position", "jobtitle"],
    "department": ["department", "dept", "functional_unit", "org_unit", "unit", "orgunit"],
    "supervisor": ["supervisor", "manager", "line_manager"],
    "email": ["email", "mail"],
    "domain": ["domain"],
}

DEPT_RULES: List[Tuple[str, str]] = [
    (r"^hr|human\s*resources?$", "HR"),
    (r"^it$|^i\.?t\.?$|information\s+technology|tech\s*ops?|systems?$", "IT"),
    (r"^engin.*|^r&d$|research\s*&?\s*development|development", "Engineering"),
    (r"^sales?$|^biz\s*dev|business\s*development|account\s*exec.*", "Sales"),
    (r"^finance$|^acct|accounting|fin(ance)?\b", "Finance"),
    (r"^legal|compliance|regulatory|general\s*counsel", "Legal"),
    (r"^marketing|communications?|public\s*relations|pr\b", "Marketing"),
    (r"^operations?$|ops\b|facilities|administration|admin(istration)?$", "Operations"),
    (r"^security|infosec|cyber|secops|csirt|soc$", "Security"),
    (r"^management|executive|c[-\s]*suite|leadership|executive\s*office", "Management"),
]

ROLE_RULES: List[Tuple[str, str]] = [
    (r"admin(istrator)?|sys\s*admin|it\s*admin|root", "IT Admin"),
    (r"manager|lead|director|head|chief|c[-\s]*\w+", "Manager"),
    (r"engineer|developer|research(er)?|scientist|analyst", "Engineer/Analyst"),
    (r"hr|human\s*resources?", "HR Staff"),
    (r"sales|account\s*(exec|manager)|business\s*dev|bd", "Sales Rep"),
    (r"legal|counsel|compliance", "Legal Staff"),
    (r"finance|accountant|controller|auditor", "Finance Staff"),
    (r"marketing|communications?|pr", "Marketing Staff"),
]


def _pick_first(colnames: List[str], candidates: List[str]) -> str:
    low = {c.lower(): c for c in colnames}
    for alias in candidates:
        if alias in low:
            return low[alias]
    for c in colnames:
        for alias in candidates:
            if alias in c.lower():
                return c
    return ""


def _norm_text(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def _norm_by_rules(val: str, rules: List[Tuple[str, str]]) -> str:
    s = _norm_text(val)
    if not s:
        return s
    s_low = s.lower()
    for pat, repl in rules:
        import re as _re
        if _re.fullmatch(pat, s_low) or _re.search(pat, s_low):
            return repl
    return s.title()


def load_ldap_dir(ldap_dir: Path) -> pd.DataFrame:
    """
    Đọc tất cả CSV trong ldap_dir, suy luận cột user/department/role/...,
    chuẩn hoá department & role, suy luận is_contractor và is_admin.
    """
    files = sorted(list(ldap_dir.rglob("*.csv")))
    if not files:
        raise SystemExit(f"[ERR] Không tìm thấy CSV trong: {ldap_dir}")
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p, dtype=str, low_memory=True)
        except Exception:
            df = pd.read_csv(p, dtype=str, low_memory=True, encoding="latin1")
        cols = list(df.columns)
        col_user = _pick_first(cols, LDAP_COLMAP["user"]) or cols[0]
        col_dept = _pick_first(cols, LDAP_COLMAP["department"])
        col_role = _pick_first(cols, LDAP_COLMAP["role"])
        col_sup = _pick_first(cols, LDAP_COLMAP["supervisor"])
        col_mail = _pick_first(cols, LDAP_COLMAP["email"])
        col_dom = _pick_first(cols, LDAP_COLMAP["domain"])

        out = pd.DataFrame()
        out["user"] = df[col_user].astype(str).str.strip().str.upper()
        if col_dept:
            out["department_raw"] = df[col_dept].map(_norm_text)
        if col_role:
            out["role_raw"] = df[col_role].map(_norm_text)
        if col_sup:
            out["supervisor"] = df[col_sup].map(_norm_text)
        if col_mail:
            out["email"] = df[col_mail].map(_norm_text)
        if col_dom:
            out["domain"] = df[col_dom].map(_norm_text)
        frames.append(out)

    ldap = pd.concat(frames, ignore_index=True)
    ldap = (
        ldap.sort_values(by=["user"])
        .groupby("user", as_index=False)
        .agg(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
    )

    def col_or_empty(df: pd.DataFrame, name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series([""] * len(df), index=df.index)

    dep_raw = col_or_empty(ldap, "department_raw")
    rol_raw = col_or_empty(ldap, "role_raw")

    ldap["department"] = dep_raw.apply(lambda x: _norm_by_rules(x, DEPT_RULES))
    ldap["role"] = rol_raw.apply(lambda x: _norm_by_rules(x, ROLE_RULES))

    # Suy luận is_contractor từ role/department
    contract_pat = r"\b(?:contract(?:or)?|extern(?:al)?|vendor|temp(?:orary)?|outsourc(?:e|ed)?)\b"
    combo_text = (rol_raw.fillna("") + " " + dep_raw.fillna("")).str.lower()
    ldap["is_contractor"] = combo_text.str.contains(
        contract_pat, regex=True, na=False
    ).astype(int)

    # is_admin từ role chuẩn hoá
    ldap["is_admin"] = ldap["role"].str.contains("Admin", case=False, na=False).astype(int)

    return ldap


def save_ldap_catalog(ldap: pd.DataFrame, out_json: Path):
    catalog = {
        "departments_detected": sorted(
            [d for d in ldap["department"].dropna().unique().tolist() if d]
        ),
        "roles_detected": sorted(
            [r for r in ldap["role"].dropna().unique().tolist() if r]
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    return catalog


# ------------------------------------------------------------
# Z-score theo user (30 ngày)
# ------------------------------------------------------------

# Những feature tuyệt đối để tính z-score
ABS_FEATURES = [
    "events",
    "logon_count",
    "failed_logon",
    "http_count",
    "file_ops",
    "file_delete_count",
    "unique_pc",
    "unique_url",
    "unique_filename",
    "after_hours_events",
    "after_hours_ratio",
    "device_ops",
    "device_filename_count",
]


def add_user_zscore(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Với mỗi user và mỗi cột trong ABS_FEATURES, tính rolling mean/std 30 ngày
    rồi sinh ra z_user_<col>_30d.
    """
    feat = feat.sort_values(["user", "date"])
    for col in ABS_FEATURES:
        if col not in feat.columns:
            feat[col] = 0.0
        mu = feat.groupby("user")[col].transform(
            lambda s: s.rolling(30, min_periods=5).mean()
        )
        sd = feat.groupby("user")[col].transform(
            lambda s: s.rolling(30, min_periods=5).std()
        )
        z = (feat[col] - mu) / sd.replace(0, np.nan)
        feat[f"z_user_{col}_30d"] = z.fillna(0.0)
    return feat


# ------------------------------------------------------------
# CLI main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        "Trích đặc trưng user×day từ logs_std.parquet (phiên bản đơn giản, ổn định cho Isolation Forest)."
    )
    ap.add_argument("--logs", required=True, help="VD: data/clear_logs/logs_std.parquet")
    ap.add_argument("--out", required=True, help="VD: artifacts/features_user_day.csv")
    ap.add_argument("--work-start", type=int, default=8)
    ap.add_argument("--work-end", type=int, default=17)
    ap.add_argument("--batch-rows", type=int, default=500_000)
    ap.add_argument("--progress-every", type=int, default=5)
    ap.add_argument("--ldap-dir", default="", help="Thư mục LDAP/*.csv (nếu có)")
    ap.add_argument("--catalog-out", default="artifacts/ldap_catalog.json")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.verbose:
        print(f"[{ts()}] Input logs = {args.logs}")

    # --- Trích feature cơ bản từ logs_std ---
    try:
        import pyarrow  # noqa
        feat_raw = process_with_pyarrow(
            args.logs,
            args.work_start,
            args.work_end,
            batch_rows=args.batch_rows,
            progress_every=args.progress_every,
            verbose=args.verbose,
        )
    except Exception as e:
        if args.verbose:
            print(f"[{ts()}] [INFO] PyArrow path failed ({e}). Fallback pandas.")
        feat_raw = process_with_pandas(
            args.logs,
            args.work_start,
            args.work_end,
            args.verbose,
        )

    feat = _finalize_basic(feat_raw)

    # --- Z-score theo user (luôn có, không phụ thuộc LDAP) ---
    feat = add_user_zscore(feat)

    # --- LDAP (nếu có) -> chỉ để thêm context: department, role, is_admin, is_contractor ---
    if args.ldap_dir:
        try:
            ldap = load_ldap_dir(Path(args.ldap_dir))
            catalog = save_ldap_catalog(ldap, Path(args.catalog_out))
            if args.verbose:
                print(f"[{ts()}] LDAP catalog saved: {args.catalog_out}")
                print("  • Departments:", ", ".join(catalog["departments_detected"]) or "(none)")
                print("  • Roles:", ", ".join(catalog["roles_detected"]) or "(none)")

            merge_cols = ["user", "department", "role", "is_admin", "is_contractor"]
            ldap_sub = ldap[[c for c in merge_cols if c in ldap.columns]].copy()
            feat = feat.merge(ldap_sub, on="user", how="left")
        except Exception as e:
            if args.verbose:
                print(f"[{ts()}] [WARN] Không thể load/merge LDAP: {e}")
            for c in ["department", "role"]:
                if c not in feat.columns:
                    feat[c] = ""
            for c in ["is_admin", "is_contractor"]:
                if c not in feat.columns:
                    feat[c] = 0
    else:
        # Không có LDAP -> đảm bảo cột tồn tại để không lỗi downstream
        for c in ["department", "role"]:
            if c not in feat.columns:
                feat[c] = ""
        for c in ["is_admin", "is_contractor"]:
            if c not in feat.columns:
                feat[c] = 0

    feat = feat.sort_values(["date", "user"]).reset_index(drop=True)
    feat.to_csv(args.out, index=False, date_format="%Y-%m-%d", encoding="utf-8-sig")

    if args.verbose:
        print(
            f"[{ts()}] Wrote features (user-day): {args.out} | "
            f"rows={len(feat):,} | users={feat['user'].nunique():,} | days={feat['date'].nunique():,}"
        )


if __name__ == "__main__":
    main()
