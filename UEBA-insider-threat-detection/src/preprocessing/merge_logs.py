"python src\merge_logs.py --in-root data_user --out-dir demo\data_clear"
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

# Cấu trúc cột tối thiểu cho từng loại, theo format CERT
MIN_COLS = {
    "logon": [
        "id", "date", "user", "pc", "activity",
        "logon_type", "ip", "logon_id",
        "success",
        "ts", "ts_round", "host", "type", "source_path",
    ],
    "file": [
        "id", "date", "user", "pc", "filename", "activity",
        "to_removable_media", "from_removable_media", "content",
        "ts", "ts_round", "host", "type", "source_path",
    ],
    "http": [
        "id", "date", "user", "pc", "url", "content",
        "ts", "ts_round", "host", "type", "source_path",
    ],
    "device": [
        "id", "date", "user", "pc", "file_tree", "activity",
        "ts", "ts_round", "host", "type", "source_path",
    ],
}

# Các cột dùng để xác định "trùng hệt nhau" cho từng loại log
DUP_COLS = {
    "logon": ["date", "user", "pc", "activity", "logon_type", "ip"],
    "file": ["date", "user", "pc", "filename", "activity"],
    "http": ["date", "user", "pc", "url", "content"],
    "device": ["date", "user", "pc", "file_tree", "activity"],
}

# ===========================
# CẤU HÌNH LỌC NOISE
# ===========================

# Các path "rác / hệ thống" thường gặp trong demo Windows
NOISY_DIRS = [
    "\\appdata\\",
    "\\temp\\",
    "\\inetcache\\",
    "\\onedrive\\logs",
    "\\packages\\microsoft.windowscommunicationsapps",
    "\\program files",
    "\\windows\\",
]

# Các thư mục người dùng "có ý nghĩa" (nếu xuất hiện thì ưu tiên giữ)
INTERESTING_DIRS = [
    r"\documents",
    r"\desktop",
    r"\downloads",
]

# Các extension file "tài liệu" phổ biến
GOOD_EXT = [
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".pdf", ".zip", ".rar", ".7z", ".txt", ".csv",
]


def read_one_csv(path: Path, dtype="str") -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype=dtype, on_bad_lines="skip")
        df["source_path"] = str(path)
        return df
    except Exception:
        return pd.DataFrame()


def ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]


def parse_time(df: pd.DataFrame) -> pd.DataFrame:
    # Ưu tiên ts (UTC ISO) -> sort theo thời gian thật
    if "ts" in df.columns:
        df["_t"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    elif "date" in df.columns:
        df["_t"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    else:
        df["_t"] = pd.NaT
    return df

def _get_first_col(df: pd.DataFrame, names: list[str]):
    """Tìm cột đầu tiên trong df khớp với danh sách tên (case-insensitive)."""
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


def normalize_logon_success(df: pd.DataFrame) -> pd.DataFrame:
    # Nếu có cột success/auth/result/status thì chuẩn hoá từ đó
    cand_cols = ["success", "auth", "result", "status", "outcome"]
    low = {c.lower(): c for c in df.columns}
    base = None
    for c in cand_cols:
        if c in low:
            base = df[low[c]].astype(str).str.strip().str.lower()
            break

    if base is None:
        # Nếu không có cột nào, suy ra từ activity
        act = df["activity"].astype(str).str.strip().str.lower()
        fail_mask = act.isin(["logonfailed", "logon_fail", "logon_failed"])
        df["success"] = np.where(fail_mask, "fail", "success")
        return df

    # Nếu có cột sẵn thì chuẩn hoá lại
    fail_mask = base.str.contains(r"fail|error|denied|reject", regex=True, na=False)
    df["success"] = np.where(fail_mask, "fail", "success")
    return df

# ===========================
# HÀM LỌC NOISE THEO TYPE
# ===========================

def _filter_file_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Giảm noise cho file logs: bỏ AppData/Temp/cache,... nếu có filename."""
    if "filename" not in df.columns:
        return df

    fn = df["filename"].astype(str).str.lower()

    # Bỏ các đường dẫn hệ thống / cache rõ ràng
    noisy_mask = pd.Series(False, index=df.index)
    for d in NOISY_DIRS:
        # THÊM regex=False Ở ĐÂY 
        noisy_mask |= fn.str.contains(d, na=False, regex=False)

    df = df[~noisy_mask]

    return df
    # Nếu muốn chặt hơn, chỉ giữ file trong thư mục user & có đuôi "đẹp"
    # nhưng để demo linh hoạt, chỉ lọc noise rõ ràng như trên là đủ.
    return df


def _filter_http_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Lọc bỏ HTTP noise cơ bản (url rỗng)."""
    if "url" not in df.columns:
        return df
    url = df["url"].astype(str).str.strip()
    df = df[url != ""]
    return df


def _filter_logon_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Lọc bớt logon noise: user rỗng / máy (kết thúc bằng $)."""
    if "user" in df.columns:
        u = df["user"].astype(str).str.strip()
        # user rỗng hoặc tài khoản máy (MACHINE$) -> bỏ
        mask_bad = (u == "") | u.str.endswith("$")
        df = df[~mask_bad]
    if "activity" in df.columns:
        a = df["activity"].astype(str).str.strip()
        df = df[a != ""]
    return df


def _filter_device_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Giữ nguyên, chỉ bỏ dòng rỗng cơ bản."""
    if "file_tree" in df.columns:
        ft = df["file_tree"].astype(str).str.strip()
        df = df[ft != ""]
    if "activity" in df.columns:
        a = df["activity"].astype(str).str.strip()
        df = df[a != ""]
    return df


def filter_logs_by_type(df: pd.DataFrame, tname: str) -> pd.DataFrame:
    """Router gọi filter phù hợp theo loại log."""
    t = tname.lower()
    if df.empty:
        return df

    if t == "file":
        return _filter_file_logs(df)
    elif t == "http":
        return _filter_http_logs(df)
    elif t == "logon":
        return _filter_logon_logs(df)
    elif t == "device":
        return _filter_device_logs(df)
    return df


def drop_dups_by_type(df: pd.DataFrame, tname: str) -> pd.DataFrame:
    """Bỏ các dòng trùng hệt nhau theo bộ cột quan trọng từng loại."""
    subset = DUP_COLS.get(tname.lower())
    if not subset:
        return df
    subset = [c for c in subset if c in df.columns]
    if not subset:
        return df
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    after = len(df)
    if before != after:
        print(f"  - dropped {before - after} duplicate rows for type={tname}")
    return df


# ===========================
# MERGE THEO LOẠI LOG
# ===========================

def merge_type(in_root: Path, out_dir: Path, tname: str) -> Path:
    # Quét TẤT CẢ file .csv dưới in_root (mọi host, mọi type)
    files = list(in_root.rglob("*.csv"))
    if not files:
        out = out_dir / f"{tname}.csv"
        pd.DataFrame(columns=MIN_COLS[tname]).to_csv(out, index=False)
        return out

    frames = []
    for p in files:
        df = read_one_csv(p)
        if df.empty:
            continue

        # Lọc đúng loại log
        if "type" in df.columns:
            df = df[df["type"].str.lower() == tname]
        else:
            parent = p.parent.name.lower()
            if parent != tname:
                continue

        if df.empty:
            continue

        # Lọc noise theo type (AppData/Temp/http rỗng/user máy,...)
        df = filter_logs_by_type(df, tname)
        if df.empty:
            continue

        df = ensure_cols(df, MIN_COLS[tname])

        if tname.lower() == "logon":
            act = df["activity"].astype(str).str.strip().str.lower()
            fail_mask = act.isin(["logonfailed", "logon_failed", "logon_fail"])
            df["success"] = np.where(fail_mask, "fail", "success")

        frames.append(df)

    if not frames:
        out = out_dir / f"{tname}.csv"
        pd.DataFrame(columns=MIN_COLS[tname]).to_csv(out, index=False)
        return out

    df_all = pd.concat(frames, ignore_index=True)

    # Lọc lần nữa & bỏ trùng toàn cục cho chắc
    df_all = filter_logs_by_type(df_all, tname)
    df_all = drop_dups_by_type(df_all, tname)

    df_all = parse_time(df_all).sort_values("_t").drop(columns=["_t"])

    out = out_dir / f"{tname}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out, index=False)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Merge incoming user CSV logs into CERT-style logon/file/http/device CSVs."
    )
    parser.add_argument(
        "--in-root",
        type=str,
        required=True,
        help="Thư mục gốc chứa data_user (vd: data_user).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Thư mục output (vd: data\\demo_logs hoặc data\\cert_logs_merged).",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=["logon", "file", "http", "device"],
        help="Các loại log cần merge (mặc định: logon file http device).",
    )

    args = parser.parse_args()
    in_root = Path(args.in_root)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    types = [t.lower() for t in args.types]
    for t in types:
        if t not in MIN_COLS:
            print(f" Bỏ qua type không hỗ trợ: {t}")
            continue
        out = merge_type(in_root, out_dir, t)
        print(f"[OK] {t} -> {out}")


if __name__ == "__main__":
    main()
