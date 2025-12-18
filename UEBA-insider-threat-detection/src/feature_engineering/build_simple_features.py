import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(
        "Rút gọn features_user_day.csv thành bản SIMPLE để train mô hình"
    )
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")

    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    if args.verbose:
        print(f"[INFO] Loaded full feature: {args.in_path}, shape={df.shape}")

    id_cols = [
        c for c in ["user", "date", "department", "role", "is_admin", "is_contractor"]
        if c in df.columns
    ]

    core = [
        "events",
        "logon_count",
        "failed_logon",
        "http_count",
        "file_ops",
        "file_delete_count",
        "after_hours_events",
        "after_hours_ratio",
        "unique_pc",
        "device_ops",
        "device_filename_count",
    ]

    zcols = [
        "z_user_events_30d",
        "z_user_logon_count_30d",
        "z_user_failed_logon_30d",
        "z_user_http_count_30d",
        "z_user_file_ops_30d",
        "z_user_after_hours_ratio_30d",
        "z_user_unique_pc_30d",
        "z_user_device_ops_30d",
        "z_user_device_filename_count_30d",
    ]
    zcols = [c for c in zcols if c in df.columns]

    keep = id_cols + core + zcols
    df_simple = df[keep].copy()

    if "events" in df_simple.columns:
        df_simple = df_simple[df_simple["events"] >= 5].reset_index(drop=True)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    df_simple.to_csv(args.out_path, index=False, encoding="utf-8-sig")

    if args.verbose:
        print(f"[INFO] Saved SIMPLE feature: {args.out_path}, shape={df_simple.shape}")


if __name__ == "__main__":
    main()
