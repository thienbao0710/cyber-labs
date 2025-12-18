"python src\ingest_gateway.py"
from flask import Flask, request, jsonify
from pathlib import Path
from datetime import datetime
import csv

app = Flask(__name__)

# --------- THƯ MỤC GỐC CHO TẤT CẢ LOG USER ---------
BASE_DIR = Path("data_user")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# Các cột core cho từng loại log
CORE_FIELDS = {
    "logon": [
        "id", "date", "user", "pc", "activity",
    ],  # + cột phụ: ts, logon_type, ip, logon_id, ...
    "file": [
        "id", "date", "user", "pc",
        "filename", "activity",
        "to_removable_media", "from_removable_media",
        "content",
    ],
    "device": [
        "id", "date", "user", "pc", "file_tree", "activity",
    ],
    "http": [
        "id", "date", "user", "pc", "url", "content",
    ],
}


@app.route("/ingest", methods=["POST"])
def ingest():
    try:
        data = request.get_json(force=True) or {}
    except Exception as e:
        print(f"[INGEST] invalid json: {e}")
        return {"status": "error", "msg": "invalid json"}, 400

    user = str(data.get("user") or "unknown")
    host = str(data.get("host") or "unknown_host")
    log_type = str(data.get("type") or "unknown")
    rows = data.get("rows") or []
    try:
        n_rows = len(rows)
    except TypeError:
        n_rows = 0

    print(f"[INGEST] user={user} host={host} type={log_type} rows={n_rows}")
    if not data:
        return jsonify({"status": "error", "msg": "no json body"}), 400

    # Thông tin cơ bản
    user = (data.get("user") or "unknown_user").strip()
    host = (data.get("host") or "unknown_host").strip()
    log_type = (data.get("type") or "").strip().lower()
    rows = data.get("rows") or []

    if not isinstance(rows, list) or len(rows) == 0:
        return jsonify({"status": "error", "msg": "no rows"}), 400

    allowed = {"logon", "http", "file", "device"}
    if log_type not in allowed:
        return jsonify({"status": "error", "msg": f"invalid type={log_type}"}), 400

    # ---------- XÁC ĐỊNH NGÀY ĐỂ ĐẶT TÊN FILE ----------
    # Ưu tiên lấy từ cột date/ts/timestamp của bản ghi, nếu không thì dùng ngày hôm nay
    file_date = None
    for r in rows:
        if not isinstance(r, dict):
            continue
        cand = r.get("date") or r.get("ts") or r.get("timestamp")
        if not cand:
            continue

        # thử nhiều format thời gian khác nhau
        for fmt in (
            "%m/%d/%Y %H:%M:%S",          # "11/16/2025 15:30:01"
            "%Y-%m-%dT%H:%M:%S",          # "2025-11-16T08:30:01"
            "%Y-%m-%dT%H:%M:%S%z",        # "2025-11-16T08:30:01+00:00"
            "%Y-%m-%dT%H:%M:%S.%fZ",      # "2025-11-16T08:30:01.123Z"
        ):
            try:
                dt = datetime.strptime(cand.split(".")[0], fmt)
                file_date = dt.date()
                break
            except Exception:
                continue
        if file_date:
            break

    if file_date is None:
        file_date = datetime.utcnow().date()

    # ---------- THƯ MỤC LƯU FILE ----------
    host_dir = BASE_DIR / host / log_type
    ensure_dir(host_dir)
    out_path = host_dir / f"{file_date.isoformat()}.csv"

    all_fields = set()
    for r in rows:
        if isinstance(r, dict):
            all_fields.update(r.keys())

    all_fields.update({"host", "user", "type"})

    core = CORE_FIELDS.get(log_type, [])
    core_present = [c for c in core if c in all_fields]
    extra = sorted(all_fields - set(core_present))

    fieldnames = core_present + extra

    file_exists = out_path.exists()

    # ---------- GHI FILE ----------
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for r in rows:
            if not isinstance(r, dict):
                continue
            row = dict(r)
            row.setdefault("host", host)
            row.setdefault("user", user)
            row.setdefault("type", log_type)
            writer.writerow(row)

    return jsonify({
        "status": "ok",
        "rows_written": len(rows),
        "host_dir": str(host_dir),
        "file": str(out_path),
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8010, debug=False)