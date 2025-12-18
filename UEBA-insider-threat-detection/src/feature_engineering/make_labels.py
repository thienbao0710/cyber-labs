# make_labels_user_day_fullv2.py
# Inputs:
#   - insiders.csv: ground-truth (user + date hoặc start/end + optional scenario_id)
#   - scenarios.txt: tiếng Anh, dạng "1. Some scenario in English..."
# Pipeline:
#   insiders -> (user,date,scenario_id)
#   scenarios.txt (EN) -> scenario_en -> VI hóa -> scenario_vi
#   scenario_vi -> infer behaviors -> behaviors + behaviors_vi
# Output (CSV):
#   user,date,label,scenario_id,behaviors,behaviors_vi
#
# Cách chạy ví dụ:
#   1) Tự dò insiders/scenarios trong folder:
#      python src\make_labels_user_day_fullv2.py ^
#        --answers data\answer ^
#        --features artifacts\features_user_day.csv ^
#        --out data\Label\cert\labels_user_day_full.csv ^
#        -v
#   2) Chỉ định file cụ thể:
#      python src\make_labels_user_day_fullv2.py ^
#        --answers . ^
#        --insiders data\answer\insiders.csv ^
#        --scenarios data\answer\scenarios.txt ^
#        --features artifacts\features_user_day.csv ^
#        --out data\Label\cert\labels_user_day_full.csv ^
#        -v

from __future__ import annotations
import argparse, glob, re
from pathlib import Path
import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True

# ---------- Robust readers ----------
_CSV_KWS = [
    dict(sep=",", engine="python", on_bad_lines="skip", low_memory=False, quotechar='"', escapechar="\\"),
    dict(sep=";", engine="python", on_bad_lines="skip", low_memory=False, quotechar='"', escapechar="\\"),
    dict(sep=",", engine="python", on_bad_lines="skip", low_memory=False),
    dict(sep=";", engine="python", on_bad_lines="skip", low_memory=False),
]
def read_csv_robust(path: str | Path) -> pd.DataFrame:
    for kw in _CSV_KWS:
        try:
            return pd.read_csv(path, **kw)
        except Exception:
            continue
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()

def read_any(path: str | Path, usecols=None) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet",".pq"):
        return pd.read_parquet(p, columns=usecols)
    return pd.read_csv(p, usecols=usecols, low_memory=False)

# ---------- Helpers ----------
def to_datestr_utc(s) -> pd.Series:
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d")

def norm_user(s) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def uniq_join(series: pd.Series) -> str | float:
    vals = [str(x).strip() for x in series.dropna().astype(str) if str(x).strip()]
    items: list[str] = []
    for v in vals:
        for p in [p.strip() for p in str(v).split(";") if p.strip()]:
            if p not in items:
                items.append(p)
    return "; ".join(items) if items else np.nan

def pick_col(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low: return low[c]
    for name in cols:
        nm = name.lower()
        if any(c in nm for c in cands):
            return name
    return None

# ---------- Parse scenarios.txt (English) ----------
def parse_scenarios_txt_en(path: str | Path) -> dict[str,str]:
    """
    Expect lines: "1. Admin installs keylogger and exfiltrates via email ..."
    Return { "1": "Admin installs keylogger and exfiltrates via email ..." }
    """
    m = {}
    p = Path(path)
    if not p.exists(): return m
    text = p.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        g = re.match(r"^\s*(\d+)\.\s*(.+)$", line)
        if g:
            m[str(int(g.group(1)))] = g.group(2).strip()
    return m

# ---------- Heuristics: scenario_vi -> behaviors ----------
BEHAVIOR_RULES = [
    ("after_hours",       [r"ngoài\s*giờ", r"late\s*night", r"outside\s*work", r"after\s*hours"]),
    ("usb_copy",          [r"\busb\b", r"thumb\s*drive", r"removable\s*drive", r"thiết\s*bị\s*usb", r"sao\s*chép\s*usb"]),
    ("data_exfil_http",   [r"\bhttp\b", r"web", r"upload", r"exfil", r"rò\s*rỉ", r"wikileaks", r"dropbox", r"google\s*drive", r"onedrive"]),
    ("data_exfil_email",  [r"email\s*home", r"email\s*cá\s*nhân", r"\bgmail\b", r"\byahoo\b", r"\boutlook\b", r"exfil.*email"]),
    ("heavy_file",        [r"file", r"sao\s*chép\s*tệp", r"copy\s*file", r"tải\s*lên\s*tệp"]),
    ("multi_pc",          [r"nhiều\s*máy", r"multiple\s*(machines|hosts|pcs)", r"dùng\s*>\s*1\s*pc", r"máy\s*người\s*khác"]),
    ("priv_misuse",       [r"đặc\s*quyền", r"privilege", r"\badmin\b", r"it\s*admin"]),
    ("malware_install",   [r"keylogger", r"malware", r"spyware", r"cài\s*mã\s*độc"]),
    ("credential_theft",  [r"keylog", r"mật\s*khẩu", r"credential"]),
    ("impersonation",     [r"giả\s*danh", r"impersonat(e|ion)", r"log\s*in\s*as"]),
    ("mass_email",        [r"email\s*hàng\s*loạt", r"mass\s*email"]),
    ("sabotage",          [r"phá\s*hoại", r"x(o|ó)a\s*(tệp|dữ\s*liệu)", r"delete\s*files", r"wipe"]),
    ("job_search",        [r"tìm\s*việc", r"job\s*site", r"career", r"recruit"]),
    ("competitor_contact",[r"đối\s*thủ", r"competitor"]),
    ("pre_departure",     [r"rời\s*công\s*ty", r"resign", r"quit", r"leave"]),
    ("trend_increase",    [r"tăng\s*dần", r"more\s*and\s*more", r"over\s*3\s*month"]),
    ("cloud_exfil",       [r"dropbox", r"google\s*drive", r"onedrive", r"cloud"]),
]

BEHAVIORS_VI_MAP = {
    "after_hours": "Hoạt động ngoài giờ",
    "usb_copy": "Sao chép USB",
    "data_exfil_http": "Rò rỉ dữ liệu qua web",
    "data_exfil_email": "Rò rỉ qua email cá nhân",
    "heavy_file": "Tăng đột biến thao tác file",
    "multi_pc": "Dùng nhiều máy tính / di chuyển ngang",
    "priv_misuse": "Lạm dụng đặc quyền",
    "malware_install": "Cài mã độc / Keylogger",
    "credential_theft": "Đánh cắp thông tin đăng nhập",
    "impersonation": "Giả danh",
    "mass_email": "Email hàng loạt",
    "sabotage": "Phá hoại / Xoá dữ liệu",
    "job_search": "Tìm việc",
    "competitor_contact": "Liên hệ đối thủ",
    "pre_departure": "Dấu hiệu trước khi nghỉ việc",
    "trend_increase": "Xu hướng tăng dần",
    "cloud_exfil": "Rò rỉ qua đám mây",
}

# ---------- English -> Vietnamese for scenario_en (heuristic) ----------
SCEN_EN2VI_RULES = [
    (r"\bafter[-\s]*hours|\blate night\b|\boutside work\b", "Hoạt động ngoài giờ"),
    (r"\busb\b|thumb drive|removable drive", "Sao chép USB"),
    (r"\bupload\b|exfil|leak|wikileaks|http|web", "Rò rỉ dữ liệu qua web"),
    (r"\bdropbox\b|google drive|onedrive|cloud", "Rò rỉ qua đám mây"),
    (r"\bemail\b|email home|personal email|gmail|yahoo|outlook", "Rò rỉ qua email cá nhân"),
    (r"\bjob\b|career|recruit", "Tìm việc"),
    (r"\bcompetitor\b", "Liên hệ đối thủ"),
    (r"\badmin\b|privilege", "Lạm dụng đặc quyền"),
    (r"\bkeylogger\b|malware|spyware", "Cài mã độc / Keylogger"),
    (r"\bimpersonat|log in as", "Giả danh"),
    (r"\bmass email\b", "Email hàng loạt"),
    (r"\bdelete\b|wipe|destroy", "Phá hoại / Xoá dữ liệu"),
    (r"\banother user'?s machine|multiple machines|many hosts", "Dùng nhiều máy tính / di chuyển ngang"),
    (r"\bsearch files|hunt data|data hunt", "Săn tìm dữ liệu"),
    (r"\bmore and more|over 3 month", "Xu hướng tăng dần"),
    (r"\bleave|resign|quit", "Dấu hiệu trước khi nghỉ việc"),
]

def scenario_en_to_vi(en_text: str) -> str:
    if not isinstance(en_text, str) or not en_text.strip():
        return ""
    vi_set = []
    low = en_text.lower()
    for pat, vi in SCEN_EN2VI_RULES:
        if re.search(pat, low):
            if vi not in vi_set:
                vi_set.append(vi)
    return "; ".join(vi_set) if vi_set else en_text  # fallback: giữ EN nếu không match

def infer_behaviors_from_vi(s: str | float) -> tuple[str | float, str | float]:
    if not isinstance(s, str) or not s.strip():
        return (np.nan, np.nan)
    hits = []
    hits_vi = []
    low = s.lower()
    for code, pats in BEHAVIOR_RULES:
        if any(re.search(p, low) for p in pats):
            hits.append(code)
            hits_vi.append(BEHAVIORS_VI_MAP.get(code, code))
    if not hits:
        return (np.nan, np.nan)
    return ("; ".join(hits), "; ".join(hits_vi))

# ---------- Core ----------
def build_labels_full(answers: str | Path, features: str | Path, out_path: str | Path,
                      insiders: str | None = None, scenarios_txt: str | None = None,
                      verbose: bool = False) -> pd.DataFrame:
    # features -> (user,date)
    feat = read_any(features)
    u_f = pick_col(feat.columns, "user"); d_f = pick_col(feat.columns, "date")
    if not u_f or not d_f:
        raise SystemExit(f"[ERR] FEATURES cần cột user/date. Thấy: {list(feat.columns)[:12]}")
    feat["user"] = norm_user(feat[u_f]); feat["date"] = to_datestr_utc(feat[d_f])
    feat = feat[["user","date"]].dropna().drop_duplicates()
    if verbose:
        print(f"[features] rows={len(feat):,} | users={feat['user'].nunique():,} | dates={feat['date'].nunique():,}")

    # auto-detect insiders/scenarios trong folder
    base_dir = Path(answers)
    if base_dir.is_dir():
        files = [Path(p) for p in glob.glob(str(base_dir / "**" / "*"), recursive=True)]
        if not insiders:
            c = [p for p in files if p.suffix.lower()==".csv" and "insiders" in p.name.lower()]
            insiders = str(c[0]) if c else None
        if not scenarios_txt:
            c = [p for p in files if p.suffix.lower()==".txt" and "scenario" in p.name.lower()]
            scenarios_txt = str(c[0]) if c else None

    if verbose:
        print(f"[answers] insiders={insiders} | scenarios_txt={scenarios_txt}")

    # load insiders
    if not insiders or not Path(insiders).exists():
        out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
        lab0 = feat.assign(label=0)
        lab0.to_csv(out, index=False, encoding="utf-8-sig")
        if verbose: print(f"[WARN] Không tìm thấy insiders -> xuất toàn 0 ({len(lab0):,})")
        return lab0
    ins = read_csv_robust(insiders)
    if ins.empty:
        out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
        lab0 = feat.assign(label=0)
        lab0.to_csv(out, index=False, encoding="utf-8-sig")
        if verbose: print(f"[WARN] insiders rỗng -> xuất toàn 0 ({len(lab0):,})")
        return lab0

    # normalize insiders
    u_i = pick_col(ins.columns, "user","userid","user_id","employee")
    if not u_i: raise SystemExit(f"[ERR] insiders thiếu cột user. Thấy: {list(ins.columns)[:12]}")
    ins["user"] = norm_user(ins[u_i])

    sc_col = pick_col(ins.columns, "scenario","scenario_id","scen")
    if sc_col and sc_col in ins.columns:
        ins["scenario_id"] = pd.to_numeric(ins[sc_col], errors="coerce").astype("Int64").astype("string")
    else:
        ins["scenario_id"] = pd.Series([pd.NA]*len(ins), dtype="string")

    date_i = pick_col(ins.columns, "date","day")
    start_i = pick_col(ins.columns, "start","begin","from")
    end_i   = pick_col(ins.columns, "end","finish","to")

    if date_i:
        base = ins[["user", date_i, "scenario_id"]].rename(columns={date_i: "date"})
        base["date"] = to_datestr_utc(base["date"])
        base = base.dropna(subset=["date"])
    elif start_i:
        ins["start_dt"] = pd.to_datetime(ins[start_i], utc=True, errors="coerce")
        if end_i:
            ins["end_dt"] = pd.to_datetime(ins[end_i], utc=True, errors="coerce")
            rng = ins.dropna(subset=["start_dt","end_dt"]).copy()
            rng["start_d"] = rng["start_dt"].dt.floor("D"); rng["end_d"] = rng["end_dt"].dt.floor("D")
            rng = rng[rng["end_d"] >= rng["start_d"]]
            exploded = []
            for _, r in rng.iterrows():
                n = int((r["end_d"] - r["start_d"]).days) + 1
                if n > 60: n = 60
                dates = pd.date_range(r["start_d"], periods=n, freq="D", tz="UTC")
                exploded.append(pd.DataFrame({
                    "user": r["user"], "date": dates,
                    "scenario_id": str(r.get("scenario_id")) if pd.notna(r.get("scenario_id")) else pd.NA
                }))
            base = pd.concat(exploded, ignore_index=True) if exploded else pd.DataFrame(columns=["user","date","scenario_id"])
            base["date"] = base["date"].dt.strftime("%Y-%m-%d")
        else:
            base = ins[["user","start_dt","scenario_id"]].rename(columns={"start_dt": "date"})
            base["date"] = base["date"].dt.strftime("%Y-%m-%d")
            base = base.dropna(subset=["date"])
    else:
        raise SystemExit(f"[ERR] insiders không có date/start. Thấy: {list(ins.columns)[:12]}")

    base = base.drop_duplicates(["user","date"])

    # scenarios: EN -> VI
    scenario_en_map = {}
    if scenarios_txt and Path(scenarios_txt).exists():
        scenario_en_map = parse_scenarios_txt_en(scenarios_txt)  # id -> en text

    def get_scen_en(sid: str) -> str:
        if pd.isna(sid) or str(sid) == "<NA>": return ""
        return scenario_en_map.get(str(sid), "")

    base["scenario_en"] = base["scenario_id"].apply(get_scen_en)
    base["scenario_vi"] = base["scenario_en"].apply(scenario_en_to_vi)

    # behaviors từ scenario_vi
    beh_codes, beh_vis = [], []
    for s in base["scenario_vi"].astype("string"):
        codes, vi = infer_behaviors_from_vi("" if s=="<NA>" else s)
        beh_codes.append(codes); beh_vis.append(vi)
    base["behaviors"] = beh_codes
    base["behaviors_vi"] = beh_vis

    base["label"] = 1

    # gộp theo (user,date)
    agg_cols = {
        "label": "max",
        "scenario_id": uniq_join,
        "behaviors": uniq_join,
        "behaviors_vi": uniq_join
    }
    base = base.groupby(["user","date"], as_index=False).agg(agg_cols)

    # merge với mọi (user,date) từ features
    lab = feat.merge(base, on=["user","date"], how="left")
    lab["label"] = lab["label"].fillna(0).astype(int)

    # ===== ONLY keep requested columns =====
    front = ["user","date","label","scenario_id","behaviors","behaviors_vi"]
    front = [c for c in front if c in lab.columns]
    lab = lab[front]

    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    lab.to_csv(out, index=False, encoding="utf-8-sig")
    if verbose:
        print(f"[OK] Wrote -> {out} | rows={len(lab):,} | positives={int(lab['label'].sum()):,} "
              f"| users={lab['user'].nunique():,} | dates={lab['date'].nunique():,}")
    return lab

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser("Make labels (insiders + scenarios EN->VI -> behaviors)")
    ap.add_argument("--answers", required=True, help="FILE or FOLDER containing insiders/scenarios")
    ap.add_argument("--features", required=True, help="artifacts/features_user_day.(csv|parquet)")
    ap.add_argument("--out", required=True, help="output labels_user_day_full.csv")
    ap.add_argument("--insiders", default=None, help="explicit insiders.csv path")
    ap.add_argument("--scenarios", default=None, help="explicit scenarios.txt path (English)")
    ap.add_argument("-v","--verbose", action="store_true")
    args = ap.parse_args()

    build_labels_full(
        answers=args.answers,
        features=args.features,
        out_path=args.out,
        insiders=args.insiders,
        scenarios_txt=args.scenarios,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
