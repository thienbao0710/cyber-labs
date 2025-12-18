# demo_api.py - Backend Flask cho HTML dashboard demo (UEBA)
# Chức năng:
#   - Serve file index.html trong html_dashboard/
#   - Serve CSV/Parquet trong thư mục demo/
#   - API POST /run_pipeline để chạy pipeline demo:
#       1) merge_incoming_csvs.py
#       2) chuan_hoa_logs.py
#       3) features_v3.py
#       4) score_iforest.py

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

from flask import Flask, jsonify, send_from_directory, request
import pandas as pd
import csv


ROOT = Path(__file__).resolve().parent

# ==============================
# Đường dẫn DEMO (chỉnh nếu thư mục bạn khác)
# ==============================
DEFAULT_DATA_USER = ROOT / "data_user"
DEFAULT_DATA_CLEAR = ROOT / "demo" / "data_clear"
DEFAULT_LOGS_STD = ROOT / "demo" / "clear_logs_demo" / "logs_std_demo.parquet"
DEFAULT_FEATURES = ROOT / "demo" / "artifacts" / "features_user_day_demo.csv"
DEFAULT_SCORES = ROOT / "demo" / "outputs" / "scores_iforest_demo.csv"
DEFAULT_MODEL = ROOT / "outputs" / "models_if_userday" / "iforest_model.joblib"
DEFAULT_LDAP_DIR = ROOT / "data_user" / "LDAP" 
DEFAULT_LDAP_CATALOG = ROOT / "demo" / "artifacts" / "ldap_catalog_demo.json"
# Lưu đánh giá SOC (demo)
DEFAULT_SOC_FEEDBACK = ROOT / "demo" / "outputs" / "soc_feedback_demo.csv"
def ensure_demo_data_ready():
    """
    Kiểm tra nếu dữ liệu demo chưa tồn tại thì chạy pipeline.
    Tránh tình trạng mở dashboard mà trống dữ liệu.
    """
    need_run = False

    files_to_check = [
        DEFAULT_FEATURES,   # features_user_day_demo.csv
        DEFAULT_SCORES,     # scores_iforest_demo.csv
        DEFAULT_LOGS_STD,   # logs_std_demo.parquet
    ]

    for f in files_to_check:
        if not Path(f).exists() or os.path.getsize(f) == 0:
            need_run = True
            break

    if need_run:
        print("\n[INIT] Không tìm thấy dữ liệu demo → TỰ ĐỘNG CHẠY PIPELINE...\n")
        ok, steps = run_full_pipeline()
        if ok:
            print("[INIT] 🎉 Pipeline chạy thành công, dữ liệu đã sẵn sàng.\n")
        else:
            print("[INIT] ❌ Pipeline lỗi – xem log chi tiết ở steps.\n")


# ==============================
# Flask app
# ==============================
app = Flask(
    __name__,
    static_folder="html_dashboard",   # chứa index.html
    static_url_path=""
)

# ==============================
# Utils: chạy subprocess
# ==============================
def run_cmd(cmd: List[str], desc: str) -> Tuple[bool, str, str]:
    """
    Chạy 1 lệnh con, trả về (ok, stdout, stderr).
    Dùng encoding='utf-8', errors='ignore' để tránh UnicodeDecodeError trên Windows.
    """
    print(f"[PIPELINE] {desc}: {' '.join(cmd)}")
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",   # quan trọng để khỏi văng UnicodeDecodeError
            env=env,
        )
        return True, completed.stdout, completed.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr
    except Exception as e:
        return False, "", str(e)

# ==============================
# Gộp 4 bước pipeline DEMO
# ==============================
def run_full_pipeline():
    """
    Pipeline demo:
      1) Merge log từ data_user → demo/data_clear
      2) Chuẩn hoá → demo/clear_logs_demo/logs_std_demo.parquet
      3) Tạo features_user_day_demo.csv
      4) Chấm điểm bằng Isolation Forest → scores_iforest_demo.csv
    """
    steps = []

    # 1) Merge log từ data_user → demo/data_clear
    cmd_merge = [
        sys.executable,
        "src/merge_incoming_csvs.py",
        "--in-root",
        str(DEFAULT_DATA_USER),
        "--out-dir",
        str(DEFAULT_DATA_CLEAR),
    ]
    ok, out, err = run_cmd(cmd_merge, "Merge log từ data_user → demo/data_clear")
    steps.append({"step": "merge_logs", "ok": ok, "stdout": out, "stderr": err})
    if not ok:
        return False, steps

    # 2) Chuẩn hoá log → logs_std_demo.parquet
    cmd_norm = [
        sys.executable,
        "src/chuan_hoa_logs.py",
        "--in-dir",
        str(DEFAULT_DATA_CLEAR),
        "--out",
        str(DEFAULT_LOGS_STD),
        "--out-format",
        "parquet",
        "-v",
    ]
    ok, out, err = run_cmd(cmd_norm, "Chuẩn hoá log (logs_std_demo.parquet)")
    steps.append({"step": "normalize_logs", "ok": ok, "stdout": out, "stderr": err})
    if not ok:
        return False, steps

    # 3) Tạo features_user_day_demo.csv
    cmd_feat = [
        sys.executable,
        "src/make_features.py",
        "--logs",
        str(DEFAULT_LOGS_STD),
        "--out",
        str(DEFAULT_FEATURES),
        "-v",
    ]

    # Nếu có thư mục LDAP cho demo thì bật LDAP
    if DEFAULT_LDAP_DIR.exists():
        cmd_feat.extend([
            "--ldap-dir",
            str(DEFAULT_LDAP_DIR),
            "--catalog-out",
            str(DEFAULT_LDAP_CATALOG),
        ])
    else:
        print(f"[PIPELINE] [WARN] Thư mục LDAP không tồn tại: {DEFAULT_LDAP_DIR} -> bỏ qua LDAP")

    ok, out, err = run_cmd(cmd_feat, "Tạo features_user_day_demo.csv")
    steps.append({"step": "features", "ok": ok, "stdout": out, "stderr": err})
    if not ok:
        return False, steps

    # 4) Chấm điểm bằng Isolation Forest → scores_iforest_demo.csv
    cmd_score = [
        sys.executable,
        "src/score_iforest.py",
        "--model",
        str(DEFAULT_MODEL),
        "--features",
        str(DEFAULT_FEATURES),
        "--out",
        str(DEFAULT_SCORES),
        "--out-format",
        "csv",
        "-v",
    ]
    ok, out, err = run_cmd(cmd_score, "Chấm điểm bằng mô hình Isolation Forest")
    steps.append({"step": "score", "ok": ok, "stdout": out, "stderr": err})

    return ok, steps
# ==============================
# SOC FEEDBACK (demo)
# ==============================

def _ensure_soc_feedback_file() -> None:
    """Đảm bảo file soc_feedback_demo.csv tồn tại và có header chuẩn."""
    if not DEFAULT_SOC_FEEDBACK.exists():
        DEFAULT_SOC_FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
        with DEFAULT_SOC_FEEDBACK.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["date", "user", "soc_tag", "soc_note"]
            )
            writer.writeheader()


@app.get("/api/soc_feedback")
def api_get_soc_feedback():
    """
    Trả về toàn bộ đánh giá SOC hiện có (demo):
      [
        {"date": "2011-01-01", "user": "USER_1", "soc_tag": "2", "soc_note": "...."},
        ...
      ]
    """
    if not DEFAULT_SOC_FEEDBACK.exists():
        return jsonify([])

    rows = []
    with DEFAULT_SOC_FEEDBACK.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return jsonify(rows)


@app.post("/api/soc_feedback")
def api_post_soc_feedback():
    """
    Lưu / cập nhật đánh giá SOC cho 1 (user, date).
    Body JSON (ví dụ):
      {
        "date": "2011-01-01",
        "user": "JSMITH",
        "soc_tag": 2,
        "soc_note": "Confirmed insider"
      }
    """
    data = request.get_json(force=True) or {}
    date = str(data.get("date", "")).strip()
    user = str(data.get("user", "")).strip().upper()
    if not date or not user:
        return jsonify({"success": False, "error": "missing date/user"}), 400

    soc_tag = int(data.get("soc_tag", 0))
    soc_note = str(data.get("soc_note") or "").strip()

    _ensure_soc_feedback_file()

    # Đọc hết, xoá dòng cũ nếu cùng (date, user)
    rows = []
    with DEFAULT_SOC_FEEDBACK.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("date") == date and str(row.get("user", "")).upper().strip() == user:
                continue
            rows.append(row)

    rows.append(
        {
            "date": date,
            "user": user,
            "soc_tag": soc_tag,
            "soc_note": soc_note,
        }
    )

    with DEFAULT_SOC_FEEDBACK.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "user", "soc_tag", "soc_note"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return jsonify({"success": True})

# ==============================
# Routes
# ==============================

@app.route("/")
def index():
    # ⭐ Khi mở dashboard sẽ check dữ liệu → tự chạy pipeline nếu thiếu
    ensure_demo_data_ready()
    return app.send_static_file("index.html")



@app.route("/demo/<path:subpath>")
def serve_demo_files(subpath: str):
    """
    Phục vụ file tĩnh trong thư mục demo/
    Ví dụ:
      /demo/artifacts/features_user_day_demo.csv
      /demo/outputs/scores_iforest_demo.csv
    """
    demo_root = ROOT / "demo"
    return send_from_directory(demo_root, subpath)

@app.get("/api/logs")
def api_logs():
    """
    Trả về tối đa 500 dòng log chuẩn hoá từ logs_std_demo.parquet / csv.
    Filter:
      - user (USER1, USER2, ...)
      - date (YYYY-MM-DD) -> lấy từ cột date nếu có, 
        không thì parse từ ts/timestamp/TIMESTAMP
      - event_type (logon/file/http/device) -> type hoặc table
    """
    user = (request.args.get("user") or "").strip().upper()
    date = (request.args.get("date") or "").strip()      # YYYY-MM-DD
    event_type = (request.args.get("event_type") or "").strip().lower()
    limit_str = request.args.get("limit", "500")

    try:
        limit = int(limit_str)
    except ValueError:
        limit = 500
    limit = max(10, min(limit, 1000))

    p = DEFAULT_LOGS_STD
    if not p.exists():
        return jsonify(
            {"success": False, "error": f"Không tìm thấy file log: {p}"}
        ), 404

    # Đọc parquet / csv
    try:
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
    except Exception as e:
        return jsonify({"success": False, "error": f"Lỗi đọc log: {e}"}), 500

    # Map tên cột lowercase -> tên thật
    colmap = {c.lower(): c for c in df.columns}

    def col(name):
        return colmap.get(name)

    mask = pd.Series(True, index=df.index)

    # Lọc user
    user_col = col("user")
    if user_col:
        df["_user_upper"] = (
            df[user_col].astype(str).str.upper().str.strip()
        )
        if user:
            mask &= df["_user_upper"] == user

    # Chuẩn hoá cột date: ưu tiên 'date', không có thì lấy từ ts/timestamp
    date_col = col("date")
    if not date_col:
        ts_candidate = col("ts") or col("timestamp") or col("timestamp")
        if ts_candidate:
            # tạo cột tạm __date_tmp dạng YYYY-MM-DD
            df["__date_tmp"] = (
                pd.to_datetime(df[ts_candidate], errors="coerce")
                  .dt.date.astype("string")
            )
            date_col = "__date_tmp"

    if date and date_col:
        mask &= df[date_col].astype(str) == date

    # Lọc event_type: type / table
    if event_type:
        type_col = col("type") or col("table") or col("event_type")
        if type_col:
            mask &= df[type_col].astype(str).str.lower() == event_type

    df = df[mask].drop(columns=["_user_upper"], errors="ignore")

    # Sắp xếp cho dễ đọc
    ts_col = col("ts") or col("timestamp") or col("timestamp")
    if ts_col:
        df = df.sort_values(ts_col, ascending=False)
    df = df.head(limit)

    rows = df.to_dict(orient="records")
    return jsonify({"success": True, "rows": rows})


@app.post("/run_pipeline")
def run_pipeline():
    """
    API để HTML gọi khi bấm nút "Chạy pipeline"
    hoặc auto 5 phút 1 lần.
    """
    ok, steps = run_full_pipeline()
    return jsonify({
        "success": ok,
        "steps": steps,
    })


if __name__ == "__main__":
    # Chạy Flask dev server
    # Mở trên trình duyệt: http://127.0.0.1:5000/
    app.run(host="0.0.0.0", port=5000, debug=True)
