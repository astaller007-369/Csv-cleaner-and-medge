"""
CSV cleaner / filler / builder (Streamlit)

1. Upload two CSV files (A and B)
2. Strip decimals -> whole numbers everywhere EXCEPT xG columns (those are never touched)
3. Copy values from one file into the other, only into blank cells (occupied cells are never overwritten),
   matching rows on date + home team + away team
4. Build a new CSV from columns of both files, in the exact order you click them
5. Optionally add percentage columns right next to the columns they are calculated from
6. Optionally delete columns, then download
"""
import csv
import hashlib
import io
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

XG_RE = re.compile(r"(^|[^a-z0-9])xg", re.I)
CHOOSE = "— choose —"
DAYFIRST = False  # set from the UI
STRICT_MATCH = True  # set from the UI: True = team names must be identical text


# ----------------------------------------------------------------------------------
# Reading / cleaning helpers (everything is kept as TEXT so nothing is altered by accident)
# ----------------------------------------------------------------------------------
def read_csv_bytes(b: bytes) -> pd.DataFrame:
    text = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = b.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        sep = csv.Sniffer().sniff(text[:5000], delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def read_cached(b: bytes) -> pd.DataFrame:
    return read_csv_bytes(b)


def clean_column(s: pd.Series, mode: str) -> pd.Series:
    """Turn a numeric text column into whole numbers. Text columns are returned untouched."""
    s = s.astype(str)
    blank = s.str.strip() == ""
    nums = pd.to_numeric(s.where(~blank), errors="coerce").astype(float)
    ok = nums.notna() & np.isfinite(nums)
    if (~blank & ~ok).any():  # contains real text -> not a number column
        return s
    intlike = s.str.fullmatch(r"\s*-?\d+\s*")
    idx = ok & ~intlike
    if not idx.any():
        return s
    v = nums[idx]
    r = np.sign(v) * np.floor(np.abs(v) + 0.5) if mode == "round" else np.trunc(v)
    out = s.copy()
    out[idx] = r.astype("int64").astype(str)
    return out


def clean_df(df: pd.DataFrame, mode: str, xg_cols) -> pd.DataFrame:
    out = df.copy()
    for c in df.columns:
        if c in xg_cols:
            continue
        out[c] = clean_column(df[c], mode)
    return out


def guess_xg(cols, extra=()):
    """Any column whose name contains xg / expected goals in any spelling: xg_home, home_xG, HomeXG, xGOT_away ..."""
    out = []
    for c in cols:
        flat = re.sub(r"[^a-z0-9]", "", c.lower())
        if "xg" in flat or "expectedgoal" in flat or any(k and k in flat for k in extra):
            out.append(c)
    return out


def canon(name):
    """Name fingerprint so home_xg, xg_home, xG_Home and HomeXG are recognised as the same column."""
    s = str(name).lower().replace("home", " home ").replace("away", " away ")
    return " ".join(sorted(re.findall(r"[a-z0-9]+", s)))


def find_target(col, tgt_cols):
    if col in tgt_cols:
        return col
    c = canon(col)
    return next((t for t in tgt_cols if canon(t) == c), None)


def guess_col(cols, names):
    low = {c.lower().strip(): i for i, c in enumerate(cols)}
    for n in names:
        if n in low:
            return low[n]
    return 0


# ----------------------------------------------------------------------------------
# Row matching (date + home team + away team)
# ----------------------------------------------------------------------------------
def norm_team(x) -> str:
    if STRICT_MATCH:
        return str(x).strip()
    s = unicodedata.normalize("NFKD", str(x))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip().lower()


def norm_date(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()
    iso = raw.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")  # year-first dates are never ambiguous
    d_iso = pd.to_datetime(raw.where(iso), errors="coerce", format="mixed", dayfirst=False)
    d_oth = pd.to_datetime(raw.where(~iso & (raw != "")), errors="coerce", format="mixed", dayfirst=DAYFIRST)
    d = d_iso.fillna(d_oth)
    out = d.dt.strftime("%Y-%m-%d")
    return out.where(d.notna(), raw).fillna("")


def make_keys(df, d, h, a, tag):
    dn = norm_date(df[d])
    hn = df[h].map(norm_team)
    an = df[a].map(norm_team)
    valid = (dn != "") & (hn != "") & (an != "")
    keys = dn + "|" + hn + "|" + an
    inv = pd.Series([f"__invalid_{tag}_{i}" for i in range(len(df))], index=df.index)
    return keys.where(valid, inv), dn


def do_transfer(src, tgt, src_keys, tgt_keys, pairs, scope, mode, tgt_xg):
    """Copy src[sc] -> tgt[tc] ONLY where tgt cell is blank. Never overwrites occupied cells."""
    tgt = tgt.copy()
    ks, _ = make_keys(src, *src_keys, "s")
    kt, _ = make_keys(tgt, *tgt_keys, "t")
    pos = pd.Series(np.arange(len(src)), index=ks)
    pos = pos[~pos.index.duplicated()]
    m = kt.map(pos)
    matched = m.notna() & scope
    idx = m[matched].astype(int).to_numpy()
    rows = []
    for sc, tc in pairs:
        incoming = pd.Series("", index=tgt.index, dtype=object)
        incoming[matched] = src[sc].to_numpy()[idx]
        blank_t = tgt[tc].astype(str).str.strip() == ""
        has = incoming.astype(str).str.strip() != ""
        fill = matched & blank_t & has
        tgt.loc[fill, tc] = incoming[fill]
        if tc not in tgt_xg:
            tgt[tc] = clean_column(tgt[tc], mode)
        rows.append({
            "from column": sc, "to column": tc,
            "cells filled": int(fill.sum()),
            "kept (already had a value)": int((matched & ~blank_t & has).sum()),
            "left blank (source blank too)": int((matched & blank_t & ~has).sum()),
        })
    summary = {
        "rows in scope": int(scope.sum()),
        "rows matched in the other file": int(matched.sum()),
        "rows with no match": int((scope & ~matched).sum()),
        "duplicate keys in source (first one used)": int(ks.duplicated().sum()),
    }
    return tgt, pd.DataFrame(rows), summary


# ----------------------------------------------------------------------------------
# New CSV builder
# ----------------------------------------------------------------------------------
def build_new(A, B, kA, kB, picks, how, sort_date):
    a, b = A.copy(), B.copy()
    a["__k"], a["__d"] = make_keys(A, *kA, "a")
    b["__k"], b["__d"] = make_keys(B, *kB, "b")
    dup = int(a["__k"].duplicated().sum()) + int(b["__k"].duplicated().sum())
    a = a[~a["__k"].duplicated()]
    b = b[~b["__k"].duplicated()]
    merged = pd.merge(a.add_prefix("A::"), b.add_prefix("B::"),
                      left_on="A::__k", right_on="B::__k", how=how).fillna("")
    alt = {}
    for i in range(3):  # date / home / away columns fall back to the other file for unmatched rows
        alt[("A", kA[i])] = ("B", kB[i])
        alt[("B", kB[i])] = ("A", kA[i])
    out, used = {}, set()
    for side, col in picks:
        ser = merged[f"{side}::{col}"].astype(str)
        if (side, col) in alt:
            o_side, o_col = alt[(side, col)]
            ser = ser.where(ser.str.strip() != "", merged[f"{o_side}::{o_col}"].astype(str))
        name = col
        if name in used:
            name = f"{col}_{side}"
        n = 2
        while name in used:
            name = f"{col}_{side}{n}"
            n += 1
        used.add(name)
        out[name] = ser.to_numpy()
    res = pd.DataFrame(out)
    if sort_date:
        d = merged["A::__d"].astype(str)
        d = d.where(d.str.strip() != "", merged["B::__d"].astype(str))
        order = d.where(d != "", "9999").to_numpy().argsort(kind="stable")
        res = res.iloc[order]
    return res.reset_index(drop=True), dup


def swap_side(name):
    if name.endswith("_home"):
        return name[:-5] + "_away"
    if name.endswith("_away"):
        return name[:-5] + "_home"
    return None


def add_percentage(df, num, den, new_name):
    n = pd.to_numeric(df[num].where(df[num].str.strip() != ""), errors="coerce").astype(float)
    d = pd.to_numeric(df[den].where(df[den].str.strip() != ""), errors="coerce").astype(float)
    pct = (n / d * 100).where(d != 0)
    pct = np.sign(pct) * np.floor(np.abs(pct) + 0.5)  # whole number, half rounds up
    col = pct.map(lambda x: "" if pd.isna(x) or not np.isfinite(x) else str(int(x)))
    out = df.copy()
    pos = max(list(out.columns).index(num), list(out.columns).index(den)) + 1
    out.insert(pos, new_name, col.to_numpy())
    return out


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# ======================================================================================
# UI
# ======================================================================================
st.set_page_config(page_title="CSV cleaner & builder", page_icon="🧮", layout="wide")
st.title("🧮 CSV cleaner, filler & builder")

u1, u2 = st.columns(2)
fA = u1.file_uploader("CSV file A", type="csv", key="fa")
fB = u2.file_uploader("CSV file B", type="csv", key="fb")
if not (fA and fB):
    st.info("Upload both CSV files to begin.")
    st.stop()

bA, bB = fA.getvalue(), fB.getvalue()
rawA, rawB = read_cached(bA), read_cached(bB)
colsA, colsB = list(rawA.columns), list(rawB.columns)

# ---------------------------------------------------------------- 1. keys + cleaning
st.header("1 · Match keys and whole-number clean-up")
k1, k2 = st.columns(2)
dA = k1.selectbox("A · date column", colsA, index=guess_col(colsA, ["date", "match_date", "kickoff"]))
hA = k1.selectbox("A · home team column", colsA, index=guess_col(colsA, ["team_home", "home_team", "home", "hometeam"]))
aA = k1.selectbox("A · away team column", colsA, index=guess_col(colsA, ["team_away", "away_team", "away", "awayteam"]))
dB = k2.selectbox("B · date column", colsB, index=guess_col(colsB, ["date", "match_date", "kickoff"]))
hB = k2.selectbox("B · home team column", colsB, index=guess_col(colsB, ["team_home", "home_team", "home", "hometeam"]))
aB = k2.selectbox("B · away team column", colsB, index=guess_col(colsB, ["team_away", "away_team", "away", "awayteam"]))
kA, kB = (dA, hA, aA), (dB, hB, aB)

DAYFIRST = st.checkbox("Dates like 10/08/2024 are day/month/year (untick for month/day/year)", value=False)
STRICT_MATCH = st.radio(
    "How strictly must the team names match between the two files?", [True, False], horizontal=True,
    format_func=lambda v: "Exact – the text must be identical (recommended)" if v
    else "Flexible – ignore capital letters, accents and extra spaces")
mode = st.radio("How should decimals be removed?", ["round", "truncate"], horizontal=True,
                format_func=lambda m: "Round to nearest whole number (0.5 and up goes up)" if m == "round"
                else "Cut the decimals off (7.9 becomes 7)")
extra_kw = st.text_input("Extra words that mark an xG column (optional, comma separated)", "")
extra = [re.sub(r"[^a-z0-9]", "", k.lower()) for k in extra_kw.split(",") if k.strip()]
fp = hashlib.md5(bA + bB).hexdigest()[:8] + extra_kw
x1, x2 = st.columns(2)
xgA = x1.multiselect("A · xG columns (left exactly as they are)", colsA, default=guess_xg(colsA, extra),
                     key=f"xgA_{fp}")
xgB = x2.multiselect("B · xG columns (left exactly as they are)", colsB, default=guess_xg(colsB, extra),
                     key=f"xgB_{fp}")

sig = hashlib.md5(bA + bB + mode.encode() + "|".join(xgA).encode() + b"||" + "|".join(xgB).encode()).hexdigest()
if st.session_state.get("sig") != sig:
    st.session_state.update(sig=sig, wA=clean_df(rawA, mode, xgA), wB=clean_df(rawB, mode, xgB),
                            new=None, report=None)
wA, wB = st.session_state["wA"], st.session_state["wB"]
st.caption(f"A: {len(wA):,} rows × {len(wA.columns)} columns · B: {len(wB):,} rows × {len(wB.columns)} columns. "
           "Changing the settings above restarts the work from the freshly cleaned files.")
with st.expander("Preview A and B"):
    st.write("A"); st.dataframe(wA.head(50), use_container_width=True)
    st.write("B"); st.dataframe(wB.head(50), use_container_width=True)

# ---------------------------------------------------------------- 2. transfer
st.header("2 · Copy data from one file into the other (blank cells only)")
direction = st.radio("Direction", ["A → B", "B → A"], horizontal=True)
a_to_b = direction.startswith("A")
src_name, tgt_name = ("A", "B") if a_to_b else ("B", "A")
src, tgt = (wA, wB) if a_to_b else (wB, wA)
src_keys, tgt_keys = (kA, kB) if a_to_b else (kB, kA)
tgt_xg = xgB if a_to_b else xgA
tag = "AB" if a_to_b else "BA"

common = [c for c in src.columns if c not in src_keys and find_target(c, list(tgt.columns))]
st.button("Select all columns that exist in both files (home_xg = xg_home etc.)", key=f"all_{tag}",
          on_click=lambda: st.session_state.__setitem__(f"cols_{tag}", common))
chosen = st.multiselect(f"Columns to copy FROM {src_name}", list(src.columns), key=f"cols_{tag}")

pairs = []
if chosen:
    st.write(f"Pick the column in {tgt_name} each one should go into:")
    grid = st.columns(3)
    opts = [CHOOSE] + list(tgt.columns)
    for i, c in enumerate(chosen):
        ft = find_target(c, list(tgt.columns))
        default = opts.index(ft) if ft else 0
        tc = grid[i % 3].selectbox(f"{src_name}: {c}  →", opts, index=default, key=f"tc_{tag}_{c}")
        pairs.append((c, tc))

scope_mode = st.radio("Which matches?", ["All teams", "One team only"], horizontal=True)
team = None
if scope_mode == "One team only":
    teams = {}
    for df_, h_, a_ in ((wA, hA, aA), (wB, hB, aB)):
        for t in pd.concat([df_[h_], df_[a_]]).unique():
            if str(t).strip():
                teams.setdefault(norm_team(t), str(t).strip())
    team = st.selectbox("Team", sorted(teams.values(), key=str.lower))

if st.button("Apply copy", type="primary", disabled=not chosen):
    if any(tc == CHOOSE for _, tc in pairs):
        st.error("Choose a target column for every selected column.")
    else:
        scope = pd.Series(True, index=tgt.index)
        if team:
            t = norm_team(team)
            scope = (tgt[tgt_keys[1]].map(norm_team) == t) | (tgt[tgt_keys[2]].map(norm_team) == t)
        new_tgt, rep, summ = do_transfer(src, tgt, src_keys, tgt_keys, pairs, scope, mode, tgt_xg)
        st.session_state["wB" if a_to_b else "wA"] = new_tgt
        st.session_state["report"] = (f"{src_name} → {tgt_name}", rep, summ)
        st.session_state["new"] = None
        st.rerun()

if st.session_state.get("report"):
    title, rep, summ = st.session_state["report"]
    st.success(f"Last copy ({title}) finished.")
    st.write(summ)
    st.caption("Rows whose date, home team AND away team do not all match exactly are ignored and nothing is copied to them.")
    st.dataframe(rep, use_container_width=True, hide_index=True)

r1, r2, r3 = st.columns(3)
r1.download_button("⬇ Download A (as it is now)", to_csv_bytes(wA), file_name="A_updated.csv", mime="text/csv")
r2.download_button("⬇ Download B (as it is now)", to_csv_bytes(wB), file_name="B_updated.csv", mime="text/csv")
if r3.button("Undo all copies (back to cleaned files)"):
    st.session_state.update(wA=clean_df(rawA, mode, xgA), wB=clean_df(rawB, mode, xgB), new=None, report=None)
    st.rerun()

# ---------------------------------------------------------------- 3. new CSV
st.header("3 · Create a new CSV from both files")
st.caption("Click columns in the order you want them. The new file follows exactly that order.")
labels = {}
for c in wA.columns:
    labels[f"A | {c}"] = ("A", c)
for c in wB.columns:
    labels[f"B | {c}"] = ("B", c)
picked = st.multiselect("Columns of the new CSV (in click order)", list(labels), key="picked")
if picked:
    st.caption("Order: " + "  →  ".join(f"{i + 1}. {p}" for i, p in enumerate(picked)))
j1, j2 = st.columns(2)
how = j1.selectbox("Rows to include", ["outer", "left", "right", "inner"],
                   format_func=lambda h: {"outer": "All rows from A and B (matched rows are combined)",
                                          "left": "All rows from A", "right": "All rows from B",
                                          "inner": "Only matches found in both"}[h])
sort_date = j2.checkbox("Sort by date, oldest first", value=True)
if st.button("Create new CSV", type="primary", disabled=not picked):
    new, dup = build_new(wA, wB, kA, kB, [labels[p] for p in picked], how, sort_date)
    st.session_state["new"] = new
    if dup:
        st.warning(f"{dup} duplicate date/home/away rows were found; the first of each was used.")

new = st.session_state.get("new")
if new is not None:
    st.success(f"New CSV ready: {len(new):,} rows × {len(new.columns)} columns")
    st.dataframe(new.head(100), use_container_width=True)

    # ------------------------------------------------------------ 4. percentages
    st.header("4 · Percentage columns (optional)")
    cols_new = list(new.columns)
    p1, p2, p3 = st.columns(3)
    num = p1.selectbox("Part (numerator)", [CHOOSE] + cols_new, key="pnum")
    den = p2.selectbox("Total (denominator)", [CHOOSE] + cols_new, key="pden")
    pname = p3.text_input("New column name", value="" if num == CHOOSE else f"{num}_percentage", key=f"pname_{num}")
    mirror = st.checkbox("Also calculate the matching _home / _away counterpart when it exists", value=True)
    if st.button("Add percentage column(s)", disabled=CHOOSE in (num, den) or not pname.strip()):
        todo = [(num, den, pname.strip())]
        if mirror:
            sn, sd = swap_side(num), swap_side(den)
            if sn in cols_new and sd in cols_new and sn and sd:
                sp = swap_side(pname.strip())
                todo.append((sn, sd, sp or f"{sn}_percentage"))
        cur, added = new, []
        for n_, d_, nm in todo:
            base, k = nm, 2
            while nm in cur.columns:
                nm = f"{base}_{k}"
                k += 1
            cur = add_percentage(cur, n_, d_, nm)
            added.append(nm)
        st.session_state["new"] = cur
        st.success("Added: " + ", ".join(added) + " (placed right after the columns they come from)")
        st.rerun()
    st.caption("Percentage = part ÷ total × 100, shown as a whole number. Blank when the total is 0 or either value is blank.")

    # ------------------------------------------------------------ 5. delete + download
    st.header("5 · Delete columns (optional) and download")
    to_del = st.multiselect("Columns to delete", list(new.columns), key="todel")
    if st.button("Delete selected columns", disabled=not to_del):
        st.session_state["new"] = new.drop(columns=to_del)
        st.rerun()
    fname = st.text_input("File name", value="new_data.csv")
    st.download_button("⬇ Download new CSV", to_csv_bytes(new), file_name=fname or "new_data.csv", mime="text/csv",
                       type="primary")
