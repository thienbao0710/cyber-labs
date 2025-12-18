"python src\chuan_hoa_logs.py --in-dir data\cert --out data\clear_logs\logs_std.parquet -v"
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re
import time
from typing import Optional, Iterable

SCHEMA_COLS = ["timestamp", "user", "pc", "table", "op","success", "filename", "bytes", "url"]
DEFAULT_TABLES = ["http", "file", "logon", "device"]
POSSIBLE_COLS = {
    "timestamp": ["timestamp", "time", "datetime", "date_time", "ts", "ts_round"],
    "date": ["date", "day"],
    "time": ["time", "hour", "hhmmss"],
    "user": ["user", "username", "employee", "user_name", "userid"],
    "pc": ["pc", "computer", "pc_name", "host", "hostname", "machine"],
    "op": ["op", "operation", "activity", "action", "event", "status"],
    "filename": ["filename", "file", "path", "filepath", "file_name"],
    "bytes": ["bytes", "size", "filesize", "nbytes"],
    "url": ["url", "uri", "link"],
    "success": ["success", "auth", "result", "status", "outcome"],
}
pd.options.mode.copy_on_write = True

def _info(msg: str): print(msg, flush=True)

def _find_first(df_cols, keys):
    s = {c.lower(): c for c in df_cols}
    for k in keys:
        if k in s: return s[k]
    return None

def _parse_timestamp(df, col_ts=None, col_date=None, col_time=None):
    def _to_local(series):
        # Parse datetime, cho phép có hoặc không timezone
        s = pd.to_datetime(series, errors="coerce")

        # Nếu có timezone (Z, +07:00, ...): convert sang Asia/Ho_Chi_Minh rồi bỏ tz
        if hasattr(s, "dt") and s.dt.tz is not None:
            s = s.dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None)

        # Nếu tz-naive: coi như giờ local, giữ nguyên
        return s

    # Khởi tạo toàn NaT
    ts = pd.Series(pd.NaT, index=df.index)

    # 1) Dùng cột ts/timestamp nếu có
    if col_ts and col_ts in df.columns:
        ts1 = _to_local(df[col_ts])
        ts = ts.combine_first(ts1)

    # 2) Nếu có cả date + time: ghép lại, fill vào chỗ còn NaT
    if (col_date in df.columns) and (col_time in df.columns):
        s2 = df[col_date].astype(str) + " " + df[col_time].astype(str)
        ts2 = _to_local(s2)
        ts = ts.combine_first(ts2)

    # 3) Nếu chỉ có date: dùng luôn cho phần còn lại
    if col_date in df.columns:
        ts3 = _to_local(df[col_date])
        ts = ts.combine_first(ts3)

    return ts




def _coerce_int(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.astype("Int64")

def normalize_chunk(chunk: pd.DataFrame, table_name: str) -> pd.DataFrame:
    cols = list(chunk.columns); low = [c.lower() for c in cols]; cmap = dict(zip(low, cols))
    c_ts   = _find_first(low, POSSIBLE_COLS["timestamp"])
    c_date = _find_first(low, POSSIBLE_COLS["date"])
    c_time = _find_first(low, POSSIBLE_COLS["time"])
    c_user = _find_first(low, POSSIBLE_COLS["user"]) or "user"
    c_pc   = _find_first(low, POSSIBLE_COLS["pc"]) or "pc"
    c_op   = _find_first(low, POSSIBLE_COLS["op"])
    c_file = _find_first(low, POSSIBLE_COLS["filename"])
    c_bytes= _find_first(low, POSSIBLE_COLS["bytes"])
    c_url  = _find_first(low, POSSIBLE_COLS["url"])
    c_success = _find_first(low, POSSIBLE_COLS.get("success", []))

    out = pd.DataFrame(index=chunk.index)
    out["timestamp"] = _parse_timestamp(chunk.rename(columns=cmap),
                                        col_ts=cmap.get(c_ts) if c_ts else None,
                                        col_date=cmap.get(c_date) if c_date else None,
                                        col_time=cmap.get(c_time) if c_time else None)
    out["user"] = chunk[cmap.get(c_user, c_user)] if (c_user in cmap or c_user in chunk.columns) else ""
    out["pc"]   = chunk[cmap.get(c_pc,   c_pc  )] if (c_pc   in cmap or c_pc   in chunk.columns) else ""
    out["table"]= table_name

    if c_op and c_op in cmap:
        op_series = chunk[cmap[c_op]].astype(str)
    else:
        op_series = pd.Series(
            {"http": "HTTP", "file": "FILE", "logon": "LOGON"}.get(table_name, "DEVICE"),
            index=chunk.index,
        )
    out["op"] = op_series.astype(str)

    if c_success and c_success in cmap:
        out["success"] = chunk[cmap[c_success]].astype(str)
    else:
        out["success"] = pd.Series("", index=chunk.index, dtype="string")

    if table_name == "logon":
        s = out["success"].astype("string").str.lower().fillna("")

        known = s.isin(["true", "false", "success", "fail", "failed", "0", "1"])

        # Một số mã lỗi logon thất bại phổ biến trong Windows Security (4625)
        fail_codes = {
            "0xc0000064",  # user name does not exist
            "0xc000006a",  # bad password
            "0xc000006d",  # bad credentials
            "0xc0000234",  # account locked
        }

        is_fail = s.isin(fail_codes)

        # Nếu là mã lỗi -> gán "fail"
        out.loc[is_fail, "success"] = "fail"

        # Nếu chưa có gì và không phải lỗi -> mặc định coi là thành công
        out.loc[(~known) & (~is_fail) & (s == ""), "success"] = "success"

    if table_name in ("file","device") and c_file and c_file in cmap:
        out["filename"] = chunk[cmap[c_file]].astype(str)
    else:
        out["filename"] = pd.Series("", index=chunk.index, dtype="string")

    if c_bytes and c_bytes in cmap: out["bytes"] = _coerce_int(chunk[cmap[c_bytes]])
    else: out["bytes"] = pd.Series(pd.array([pd.NA]*len(chunk), dtype="Int64"))

    if table_name=="http" and c_url and c_url in cmap: out["url"] = chunk[cmap[c_url]].astype(str)
    else: out["url"] = pd.Series("", index=chunk.index, dtype="string")

    out["user"] = out["user"].astype("string").str.strip().str.upper()
    out["pc"]   = out["pc"].astype("string").str.strip().str.upper()
    out = out.dropna(subset=["timestamp"])
    out = out[out["user"] != ""]
    return out[SCHEMA_COLS]

def find_input_file(in_dir: Path, table_name: str) -> Optional[Path]:
    exact = in_dir / f"{table_name}.csv"
    if exact.exists(): return exact
    cands = [p for p in in_dir.rglob("*.csv") if table_name in p.name.lower()]
    cands.sort()
    return cands[0] if cands else None

class Writer:
    def __init__(self, out_path: Path, out_format: str):
        self.out_path = out_path
        self.out_format = out_format
        self.tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        self._parquet_writer = None
        self._csv_header_written = False
        if out_format == "parquet":
            try:
                import pyarrow as pa, pyarrow.parquet as pq  # noqa
                self._has_pa = True
            except Exception:
                _info("[WARN] Không có pyarrow -> fallback CSV")
                self.out_format = "csv"; self._has_pa = False
        else:
            self._has_pa = False
        try:
            if self.tmp_path.exists(): self.tmp_path.unlink()
        except Exception:
            pass

    def write(self, df: pd.DataFrame):
        if df is None or df.empty: return
        if self.out_format=="parquet": self._write_parquet(df)
        else: self._write_csv(df)

    def _write_parquet(self, df: pd.DataFrame):
        import pyarrow as pa, pyarrow.parquet as pq
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._parquet_writer is None:
            self._parquet_writer = pq.ParquetWriter(
                where=self.tmp_path.as_posix(), schema=table.schema, compression="snappy"
            )
        self._parquet_writer.write_table(table)

    def _write_csv(self, df: pd.DataFrame):
        df.to_csv(self.tmp_path, mode="a", index=False,
                  header=not self._csv_header_written, encoding="utf-8-sig")
        self._csv_header_written = True

    def close(self):
        if self._parquet_writer is not None: self._parquet_writer.close()
        if self.tmp_path.exists():
            if self.out_path.exists():
                try: self.out_path.unlink()
                except Exception: pass
            self.tmp_path.rename(self.out_path)

def safe_read_csv(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    try:
        return pd.read_csv(path, chunksize=chunksize, dtype=str, encoding="utf-8",
                           low_memory=True, on_bad_lines="skip")
    except Exception: pass
    try:
        f = open(path, "r", encoding="utf-8", errors="ignore", newline="")
        return pd.read_csv(f, chunksize=chunksize, dtype=str, low_memory=True, on_bad_lines="skip")
    except Exception: pass
    return pd.read_csv(path, chunksize=chunksize, dtype=str, encoding="latin1",
                       engine="python", low_memory=True, on_bad_lines="skip")

def process_one_table(csv_path: Path, table_name: str, writer: Writer,
                      chunksize: int, verbose: bool, log_every: int = 5):
    if verbose: _info(f"[..] Đang đọc {table_name}: {csv_path}")
    it = safe_read_csv(csv_path, chunksize)
    total_rows = 0; chunks = 0; t0 = time.time(); last_log = t0
    for chunk in it:
        chunks += 1
        df = normalize_chunk(chunk, table_name)
        n = len(df)
        if n: writer.write(df); total_rows += n
        now = time.time()
        if (chunks % log_every == 0) or (now - last_log >= 5):
            speed = total_rows / max(now - t0, 1)
            _info(f"[{table_name:<6}] chunk={chunks:>5}  +{n:>8,}  total={total_rows:>12,}  ~{int(speed):,} rows/s")
            last_log = now
    if verbose:
        dur = time.time()-t0; speed = total_rows / max(dur,1)
        _info(f"[OK] {table_name}: {total_rows:,} dòng, {chunks} chunk, {dur:.1f}s, ~{int(speed):,} rows/s")

def main():
    ap = argparse.ArgumentParser("Chuẩn hoá logs về 1 file hợp nhất")
    ap.add_argument("--in-dir", required=True, help="Thư mục chứa các file .csv")
    ap.add_argument("--out", required=True, help="Đường dẫn output (.parquet hoặc .csv)")
    ap.add_argument("--out-format", choices=["parquet","csv"], default=None)
    ap.add_argument("--tables", default="http,file,logon,device")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out)
    out_format = (args.out_format or out_path.suffix.lower().lstrip(".") or "parquet").lower()
    tables = [t.strip().lower() for t in re.split(r"[,\s]+", args.tables) if t.strip()]
    if not in_dir.exists(): raise SystemExit(f"[ERR] Không tìm thấy thư mục: {in_dir}")

    files = {}
    for t in DEFAULT_TABLES:
        if t not in tables: continue
        p = find_input_file(in_dir, t)
        if p is None: _info(f"[WARN] Không tìm thấy file cho bảng '{t}' trong {in_dir} -> bỏ qua")
        else: files[t] = p
    if not files: raise SystemExit("[ERR] Không có file đầu vào nào.")

    writer = Writer(out_path=out_path, out_format=out_format)
    for t in DEFAULT_TABLES:
        if t not in files: continue
        process_one_table(files[t], t, writer, args.chunksize, args.verbose, args.log_every)
    writer.close()
    if out_path.exists():
        msg = f"[DONE] Hợp nhất -> {out_path}"
        if out_format=="csv":
            try:
                with out_path.open("r", encoding="utf-8-sig", errors="ignore") as f:
                    line_count = sum(1 for _ in f)
                msg += f"  rows≈{line_count-1:,}"
            except Exception: pass
        _info(msg)
    else:
        _info("[ERR] Không tạo được output")

if __name__ == "__main__":
    main()
