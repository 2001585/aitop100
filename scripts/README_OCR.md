# OCR + Analysis Guide

This repo contains weekly menu images under `ai_top_100_menu/menu_images/` and a script that OCRs and analyzes them to answer the questions in `문제사진/1.md`.

## 1) Install OCR tooling

Ubuntu / WSL (recommended):

```
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-kor python3-pip
python3 -m pip install -r scripts/requirements.txt
```

macOS (Homebrew):

```
brew install tesseract
brew install tesseract-lang  # or: brew install tesseract-lang --all-languages
pip3 install -r scripts/requirements.txt
```

Windows (winget + pip):

1) Install Tesseract (choose "Korean" during setup) from: https://github.com/UB-Mannheim/tesseract/wiki

2) Then:

```
python -m pip install -r scripts/requirements.txt
```

Verify:

```
tesseract --version
python3 - << 'PY'
import pytesseract, PIL
print('pytesseract OK')
PY
```

## 2) Run OCR + analysis

```
python3 scripts/ocr_menu.py
```

Output: `build/menu_data.json` with structured data and the computed summaries for Q1–Q5.

Key fields:

- `summary.Q1_order`: Suffix frequency order among `조림/볶음/무침/구이` for 2025‑01‑13 week (lunch, specified corners)
- `summary.Q2_avg_order`: January lunch average kcal order among `한식A/한식B/양식/팝업A/팝업B`
- `summary.Q3_regions_2plus`: Regions appearing 2+ times across Jan–Feb
- `summary.Q4_order` and `summary.Q4_menu_kcal`: Calorie order and individual kcal for target menus
- `summary.Q5_solution`: Array of objects in the requested answer format for all days in February

If Tesseract recognition is imperfect on your machine, re-run with better lighting/ DPI or try:

```
PYTESSERACT_TESSERACT_CMD=/usr/bin/tesseract \
python3 scripts/ocr_menu.py
```

You can also tweak OCR config inside `scripts/ocr_menu.py` (psm/oem) if needed.

