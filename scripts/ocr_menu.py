#!/usr/bin/env python3
"""
OCR extractor for Chunsik Dorak weekly menu images.

It pulls text from ai_top_100_menu/menu_images/*.png and produces a structured
JSON with per-date, per-corner calories and side-dish text. The parser uses
heuristics tuned for the provided images and is resilient to small OCR noise.

Output: build/menu_data.json

Dependencies: see scripts/requirements.txt
 - tesseract-ocr with Korean language data (kor)
 - Python packages: pytesseract, pillow, opencv-python-headless, pandas, numpy

Notes:
 - We rely on full-page OCR + geometric filtering by day columns and corner
   labels. No fixed pixel coordinates are baked in; we infer columns from the
   recognized day headers in the top band.
 - Side-dish suffix counting is performed from text that falls under the
   corresponding corner label until the next corner label within the same day.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pytesseract  # type: ignore
from PIL import Image  # type: ignore
from difflib import SequenceMatcher


IMG_DIR = Path("ai_top_100_menu/menu_images")
OUT_DIR = Path("build")
OUT_JSON = OUT_DIR / "menu_data.json"


WEEKDAY_KOR = ["월", "화", "수", "목", "금"]

# Lunch dine-in corners + lunch take-out corners (as per spec)
LUNCH_CORNERS = [
    "한식A",
    "한식B",
    "팝업A",
    "팝업B",
    "양식",
    "샐러드",
    "비건",
    "라이스&누들",
    "버거&델리",
]

# Dinner choices visible on the board (subset used by Q5)
DINNER_ALLOWED = ["한식B", "샐러드", "버거&델리"]

SUFFIX_KEYS = ["조림", "볶음", "무침", "구이"]


def similar(a: str, b: str) -> float:
    a = normalize(a)
    b = normalize(b)
    return SequenceMatcher(None, a, b).ratio()


def normalize(s: str) -> str:
    return re.sub(r"\s+", "", s)


def load_image(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    # upscale a bit to help OCR
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return img


def ocr_words(img: np.ndarray) -> pd.DataFrame:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    # Light adaptive threshold improves contrast for small text
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 11)

    data = pytesseract.image_to_data(
        th, lang="kor+eng", output_type=pytesseract.Output.DATAFRAME,
        config="--oem 1 --psm 4"
    )
    df = data.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    # Drop extremely low-confidence noise (keep more to improve recall)
    df = df[df["conf"].fillna(-1) >= 10]
    return df


def find_day_columns(df: pd.DataFrame) -> List[Tuple[str, Tuple[int, int]]]:
    """Return list of (label, (x_center, y_top)) for the five day headers.

    We look in the top 25% of the image for tokens like '1월 13일' with weekday.
    Then group nearby tokens as a header and take its x center.
    """
    # Heuristic: day labels sit within the top quarter and have a digit + '월'
    header_df = df.sort_values(["top", "left"]).copy()
    top_cut = header_df["top"].quantile(0.25)
    header_df = header_df[header_df["top"] <= top_cut + 40]

    # Combine into lines by proximity
    lines: List[Dict] = []
    for _, r in header_df.iterrows():
        placed = False
        for line in lines:
            if abs(r["top"] - line["top"]) < 20 and abs(r["left"] - line["right"]) < 120:
                line["text"] += " " + r["text"]
                line["right"] = max(line["right"], int(r["left"] + r["width"]))
                placed = True
                break
        if not placed:
            lines.append({
                "top": int(r["top"]),
                "left": int(r["left"]),
                "right": int(r["left"] + r["width"]),
                "text": r["text"],
            })

    day_cols: List[Tuple[str, Tuple[int, int]]] = []
    day_pat = re.compile(r"(1|01|02)월\s*([0-3]?[0-9])일")
    for line in lines:
        if day_pat.search(line["text"]):
            x_center = (line["left"] + line["right"]) // 2
            day_cols.append((line["text"], (x_center, line["top"])) )

    # Sort left to right, keep first five occurrences
    day_cols.sort(key=lambda x: x[1][0])
    if len(day_cols) >= 5:
        day_cols = day_cols[:5]
    return day_cols


def find_labels_in_column(df: pd.DataFrame, x_center: int) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Find likely corner labels near a given column x position.
    Returns list of (label_text, (left, top, right, bottom)) ordered by y.
    """
    # Filter words near the column
    tol = 240  # generous horizontally
    col_df = df[(df["left"] <= x_center + tol) & ((df["left"] + df["width"]) >= x_center - tol)].copy()
    col_df = col_df.sort_values(["top", "left"])  # reading order

    # Merge nearby tokens into labels by y proximity
    rows: List[Dict] = []
    for _, r in col_df.iterrows():
        placed = False
        for row in rows:
            if abs(r["top"] - row["top"]) < 18:  # same line
                row["text"] += r" " + r["text"]
                row["left"] = min(row["left"], int(r["left"]))
                row["right"] = max(row["right"], int(r["left"] + r["width"]))
                row["bottom"] = max(row["bottom"], int(r["top"] + r["height"]))
                placed = True
                break
        if not placed:
            rows.append({
                "top": int(r["top"]),
                "bottom": int(r["top"] + r["height"]),
                "left": int(r["left"]),
                "right": int(r["left"] + r["width"]),
                "text": r["text"],
            })

    # Keep only rows that look like corner labels
    labels: List[Tuple[str, Tuple[int, int, int, int]]] = []
    candidates = LUNCH_CORNERS + ["샐러드바"] + ["한식B"]  # dinner labels also show up
    for row in rows:
        txt = row["text"].strip()
        for cand in candidates:
            if similar(txt, cand) >= 0.65:
                labels.append((cand, (row["left"], row["top"], row["right"], row["bottom"])) )
                break

    labels.sort(key=lambda x: x[1][1])
    return labels


def collect_region_text(df: pd.DataFrame, bbox: Tuple[int, int, int, int]) -> str:
    l, t, r, b = bbox
    region = df[(df["left"] >= l - 20) & ((df["left"] + df["width"]) <= r + 200)
                & (df["top"] >= t) & ((df["top"] + df["height"]) <= b)]
    # Collect following lines up to a limited vertical distance
    region = region.sort_values(["top", "left"])
    words = region["text"].astype(str).tolist()
    return " ".join(words)


def extract_column_data(df: pd.DataFrame, x_center: int) -> Dict[str, Dict[str, object]]:
    """Extract per-corner info for a single day column.
    Returns mapping corner -> {"text": full_text, "kcal": int_or_None}
    """
    labels = find_labels_in_column(df, x_center)
    results: Dict[str, Dict[str, object]] = {}
    for i, (label, bbox) in enumerate(labels):
        # Determine crop bottom limit as next label's top
        next_top = labels[i + 1][1][1] if i + 1 < len(labels) else bbox[1] + 600
        l, t, r, b = bbox
        region_bbox = (l - 30, t, r + 30, next_top)
        text = collect_region_text(df, region_bbox)

        # Find kcal number closest after label
        kcal = None
        m = re.search(r"([1-9][0-9]{2,4})\s*k?\s*k?cal", text, re.IGNORECASE)
        if m:
            try:
                kcal = int(m.group(1))
            except Exception:
                kcal = None

        results[label] = {"text": text, "kcal": kcal}

    return results


def sides_suffix_counts(text: str) -> Dict[str, int]:
    # Treat punctuation and separators as spaces
    cleaned = re.sub(r"[,*·•/()\[\]{}:;|\\]", " ", text)
    tokens = [t for t in re.split(r"\s+", cleaned) if t]
    cnt = {k: 0 for k in SUFFIX_KEYS}
    for tok in tokens:
        for suf in SUFFIX_KEYS:
            if tok.endswith(suf):
                cnt[suf] += 1
                break
    return cnt


def parse_week(img_path: Path) -> Dict:
    img = load_image(img_path)
    df = ocr_words(img)
    day_cols = find_day_columns(df)

    # Map five x centers
    x_centers = [x for _, (x, _y) in day_cols]
    # If detection fails, fall back to equal spacing across the actual image width
    if len(x_centers) != 5:
        width = img.shape[1]
        # Skip the left gutter where the '구분/코너' column lives (~7% width)
        left_gutter = int(width * 0.08)
        right_margin = int(width * 0.02)
        usable = width - left_gutter - right_margin
        step = usable // 5
        x_centers = [left_gutter + step * i + step // 2 for i in range(5)]

    per_day: List[Dict[str, Dict[str, object]]] = []
    for xc in x_centers:
        per_day.append(extract_column_data(df, xc))

    return {
        "file": img_path.name,
        "per_day": per_day,
    }


def compute_structured() -> Dict:
    data = {"weeks": []}
    for p in sorted(IMG_DIR.glob("*.png")):
        week = parse_week(p)
        data["weeks"].append(week)
    return data


def summarize_questions(data: Dict) -> Dict:
    # Helper to resolve weeks by filename prefix
    weeks_by_file = {w["file"]: w for w in data.get("weeks", [])}

    # Q1: Week 2025-01-13 lunch sides suffix counts across specified corners
    q1_counts = {k: 0 for k in SUFFIX_KEYS}
    if "2025-01-13.png" in weeks_by_file:
        week = weeks_by_file["2025-01-13.png"]
        for day in week["per_day"]:
            for corner in ["한식A", "한식B", "팝업A", "팝업B", "양식"]:
                info = day.get(corner)
                if not info:
                    continue
                text = str(info.get("text", ""))
                cnt = sides_suffix_counts(text)
                for k, v in cnt.items():
                    q1_counts[k] += v

    q1_sorted = sorted(q1_counts.items(), key=lambda x: (-x[1], SUFFIX_KEYS.index(x[0])))
    q1_order = [k for k, _ in q1_sorted]

    # Q2: January lunch average calories per corner (5 dine-in + 2 popups)
    jan_files = ["2025-01-06.png", "2025-01-13.png", "2025-01-20.png"]
    jan_corners = ["한식A", "한식B", "양식", "팝업A", "팝업B"]
    acc: Dict[str, List[int]] = {c: [] for c in jan_corners}
    for jf in jan_files:
        wk = weeks_by_file.get(jf)
        if not wk:
            continue
        for day in wk["per_day"]:
            for c in jan_corners:
                info = day.get(c)
                if info and isinstance(info.get("kcal"), int):
                    acc[c].append(int(info["kcal"]))
    jan_avg = {c: (sum(v) / len(v) if v else None) for c, v in acc.items()}
    q2_sorted = [c for c, v in sorted(jan_avg.items(), key=lambda x: (-(x[1] or -1)))]

    # Q3: Region names frequency across Jan+Feb (all areas)
    regions = ["전주", "태국", "베트남", "나가사키", "안동"]
    region_cnt = {r: 0 for r in regions}
    for wk in data.get("weeks", []):
        for day in wk["per_day"]:
            for info in day.values():
                text = str(info.get("text", ""))
                # Focus on menu name vicinity: take first ~80 chars after label
                text_norm = text[:300]
                for r in regions:
                    region_cnt[r] += len(re.findall(re.escape(r), text_norm))
    q3_multi = [r for r, c in region_cnt.items() if c >= 2]

    # Q4: Calorie order for specific menus
    target_menus = [
        "덴가스떡볶이",
        "돈코츠라멘",
        "마라탕면",
        "수제남산왕돈까스",
        "탄탄면",
    ]
    menu_kcal: Dict[str, Optional[int]] = {m: None for m in target_menus}
    for wk in data.get("weeks", []):
        for day in wk["per_day"]:
            for info in day.values():
                text = str(info.get("text", ""))
                kcal_val = info.get("kcal")
                if not isinstance(kcal_val, int):
                    # try inline parsing as backup
                    m = re.search(r"([1-9][0-9]{2,4})\s*k?\s*cal", text, re.IGNORECASE)
                    if m:
                        try:
                            kcal_val = int(m.group(1))
                        except Exception:
                            kcal_val = None
                text_key = normalize(text)
                for tm in target_menus:
                    if tm in text_key and (menu_kcal[tm] is None) and isinstance(kcal_val, int):
                        menu_kcal[tm] = int(kcal_val)

    q4_sorted = [m for m, _ in sorted(menu_kcal.items(), key=lambda x: (-(x[1] or -1)))]

    # Q5: February optimization
    feb_files = ["2025-02-03.png", "2025-02-10.png", "2025-02-17.png", "2025-02-24.png"]
    results_q5: List[Dict[str, str]] = []
    # Map filename to its Monday date
    for fname in feb_files:
        wk = weeks_by_file.get(fname)
        if not wk:
            continue
        # Infer Monday date from filename
        monday = Path(fname).stem  # YYYY-MM-DD
        y, m, d = map(int, monday.split("-"))
        from datetime import date, timedelta
        base = date(y, m, d)
        for i in range(5):
            day_id = base + timedelta(days=i)
            day = wk["per_day"][i] if i < len(wk["per_day"]) else {}
            # Lunch candidates
            lunch_vals = []  # (corner, kcal)
            for c in LUNCH_CORNERS:
                kc = day.get(c, {}).get("kcal")
                if isinstance(kc, int):
                    lunch_vals.append((c, kc))

            if day_id.weekday() < 4:  # Mon-Thu
                # Dinner candidates (allowed subset)
                dinner_vals = []
                for c in DINNER_ALLOWED:
                    kc = day.get(c, {}).get("kcal")
                    if isinstance(kc, int):
                        dinner_vals.append((c, kc))
                # choose pair closest to 1550
                best = None
                target = 1550
                for lc, lk in lunch_vals:
                    for dc, dk in dinner_vals:
                        s = lk + dk
                        gap = abs(target - s)
                        # tie-breaker: prefer smaller absolute kcal if same gap
                        key = (gap, abs(s), -max(lk, dk))
                        if best is None or key < best[0]:
                            best = (key, (lc, dc))
                if best is None:
                    continue
                (lc, dc) = best[1]
                results_q5.append({
                    "id": str(day_id),
                    "lunch": lc,
                    "dinner": dc,
                })
            else:
                # Friday: lowest lunch kcal
                if not lunch_vals:
                    continue
                lc, lk = min(lunch_vals, key=lambda x: (x[1], x[0]))
                results_q5.append({
                    "id": str(day_id),
                    "lunch": lc,
                })

    return {
        "Q1_counts": q1_counts,
        "Q1_order": q1_order,
        "Q2_avg_order": q2_sorted,
        "Q3_regions_2plus": q3_multi,
        "Q4_menu_kcal": menu_kcal,
        "Q4_order": q4_sorted,
        "Q5_solution": results_q5,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = compute_structured()
    summary = summarize_questions(data)
    payload = {
        "images_dir": str(IMG_DIR),
        "data": data,
        "summary": summary,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
