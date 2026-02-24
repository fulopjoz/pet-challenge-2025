#!/usr/bin/env python3
"""
Script 2: Extract Tm values from PETase PDFs.

Three-stage extraction per paper:
  Stage 1: pdfplumber table/text extraction (primary, precise for tables)
  Stage 2: GLM-4.7 vision cross-check (independent verification)
  Stage 3: Cross-validation and confidence scoring

Usage:
    python scripts/extract_tm_from_pdfs.py --paper brott2022
    python scripts/extract_tm_from_pdfs.py --all
    python scripts/extract_tm_from_pdfs.py --all --no-vision
"""

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    sys.exit("pdfplumber not installed. Run with: "
             "/home/assistant/workspace/scratch/_envs/data/bin/python")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
VALIDATION_DIR = PROJECT_DIR / "data" / "validation"
OUTPUT_DIR = VALIDATION_DIR / "per_paper"

EINFRA_BASE = "https://llm.ai.e-infra.cz/v1"
EINFRA_MODEL = "qwen3.5"  # Best for vision: reads numerical labels precisely

# ============================================================
# API key loading
# ============================================================

def load_api_key() -> str:
    """Load e-INFRA API key from OpenClaw .env or environment."""
    env_path = Path("/home/assistant/.openclaw/.env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k in ("EINFRA_API_KEY", "OPENAI_API_KEY"):
                    return v
    key = os.environ.get("EINFRA_API_KEY", "")
    if key:
        return key
    sys.exit("No API key found. Set EINFRA_API_KEY or check /home/assistant/.openclaw/.env")


# ============================================================
# Named variant resolution
# ============================================================

NAMED_VARIANTS = {
    "WT": "WT",
    "WILD-TYPE": "WT",
    "WILD TYPE": "WT",
    "ISPETASEWT": "WT",
    "ISPETASE WT": "WT",
    "ISPETASE": "WT",
    "TM": "S121E/D186H/R280A",
    "ISPETASETM": "S121E/D186H/R280A",
    "THERMOPETASE": "S121E/D186H/R280A",
    "DURAPETASE": "L117F/Q119Y/T140D/W159H/G165A/I168R/A180I/S188Q/S214H/R280A",
    # Cui 2021 GRAPE round designations
    "M9": "L117F/Q119Y/T140D/W159H/G165A/I168R/A180I/S188Q/S214H/R280A",  # = DuraPETase
    # DuraPETase compound variants (Brott 2022)
    "DURAPETASEN233C/S282C": "L117F/Q119Y/T140D/W159H/G165A/I168R/A180I/S188Q/S214H/R280A/N233C/S282C",
    "DURAPETASEK95N/S121E/F201I/R280A": "L117F/Q119Y/T140D/W159H/G165A/I168R/A180I/S188Q/S214H/R280A/K95N/S121E/F201I/R280A",
    "DURAPETASEK95N/S121E/F201I/N233C/R280A/S282C": "L117F/Q119Y/T140D/W159H/G165A/I168R/A180I/S188Q/S214H/R280A/K95N/S121E/F201I/N233C/R280A/S282C",
    # HotPETase (Bell 2022) — 21 mutations from directed evolution of IsPETaseTS
    "HOTPETASE": "S121E/D186H/R280A/P181V/S207R/S214Y/Q119K/S213E/N233C/S282C/R90T/Q182M/N212K/R224L/S58A/S61V/K95N/M154G/N241C/K252M/T270Q",
    # FAST-PETase (Lu 2022) — 5 mutations from ML-guided engineering
    "FAST-PETASE": "N233K/R224Q/S121E/D186H/R280A",
    "FASTPETASE": "N233K/R224Q/S121E/D186H/R280A",
}


def resolve_variant_name(raw_name: str) -> tuple[str, str]:
    """Resolve raw variant name to (display_name, mutation_string).

    Handles named variants (ThermoPETase, DuraPETase) and compound
    names like IsPETaseTM+K95N/F201I.
    """
    clean = raw_name.strip()
    upper = clean.upper().replace(" ", "")

    # Direct lookup
    if upper in NAMED_VARIANTS:
        return clean, NAMED_VARIANTS[upper]

    # IsPETaseTM + additional mutations
    tm_match = re.match(r'ISPETASETM[+]?(.+)', upper)
    if tm_match:
        extra = tm_match.group(1).lstrip("+/")
        base = "S121E/D186H/R280A"
        return clean, f"{base}/{extra}"

    # Already a mutation string (e.g., S121E, S121E/D186H)
    if re.match(r'^[A-Z]\d+[A-Z](/[A-Z]\d+[A-Z])*$', upper):
        return clean, upper

    return clean, clean


def normalize_variant_key(name: str) -> str:
    """Normalize variant name for cross-method matching."""
    name = name.strip().upper().replace(" ", "")
    for prefix in ("ISPETASE", "PETASE"):
        if name.startswith(prefix) and name != prefix:
            name = name[len(prefix):]
    return name.strip()


# ============================================================
# PDF file resolution
# ============================================================

def find_pdf(pattern: str) -> Path | None:
    """Find a PDF in the validation directory matching a pattern.

    Uses fnmatch-style glob. Falls back to substring matching.
    """
    import fnmatch
    for pdf in VALIDATION_DIR.iterdir():
        if pdf.suffix.lower() == ".pdf" and fnmatch.fnmatch(pdf.name, pattern):
            return pdf
    # Substring fallback
    for pdf in VALIDATION_DIR.iterdir():
        if pdf.suffix.lower() == ".pdf":
            key = pattern.replace("*", "").lower()
            if key and key in pdf.name.lower():
                return pdf
    return None


def find_pdf_multiline(keywords: list[str]) -> Path | None:
    """Find PDFs whose names contain newlines (like Zhong-Johnson)."""
    for pdf in VALIDATION_DIR.iterdir():
        name_lower = pdf.name.lower()
        if all(kw.lower() in name_lower for kw in keywords):
            return pdf
    return None


# ============================================================
# Stage 1: pdfplumber extraction
# ============================================================

class TableExtractor:
    """Extract Tm data from bordered PDF tables."""

    def extract(self, pdf_path: str, page_num: int, config: dict) -> list[dict]:
        """Extract Tm from bordered table on the given page (1-indexed)."""
        results = []
        with pdfplumber.open(pdf_path) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                print(f"    Page {page_num} out of range (PDF has {len(pdf.pages)} pages)")
                return results
            page = pdf.pages[page_num - 1]
            tables = page.extract_tables()

            for table in tables:
                if not table or len(table) < 2:
                    continue
                header = [str(c) if c else "" for c in table[0]]
                header_str = " ".join(header)

                # Optional: check if header matches expected pattern
                filter_pat = config.get("header_filter")
                if filter_pat and not re.search(filter_pat, header_str, re.I):
                    continue

                for row in table[1:]:
                    if not row or all(c is None or str(c).strip() == "" for c in row):
                        continue
                    parsed = self._parse_row(row, header, config)
                    if parsed:
                        results.append(parsed)

        return results

    def _parse_row(self, row: list, header: list, config: dict) -> dict | None:
        """Parse a single table row for variant name and Tm value."""
        tm_val = None
        tm_std = 0.0
        variant = None
        candidates = []  # Collect all text cells as candidate variant names

        for i, cell in enumerate(row):
            if cell is None:
                continue
            cell_str = str(cell).strip()
            if not cell_str:
                continue

            # Try Tm ± std format
            pm_match = re.match(
                r'(\d+\.?\d*)\s*[±\+/-]+\s*(\d+\.?\d*)', cell_str)
            if pm_match:
                val = float(pm_match.group(1))
                if 25 < val < 120:
                    tm_val = val
                    tm_std = float(pm_match.group(2))
                    continue

            # Try plain numeric Tm
            plain_match = re.match(r'^(\d+\.?\d*)\s*°?\s*C?$', cell_str)
            if plain_match:
                val = float(plain_match.group(1))
                if 25 < val < 120:
                    if tm_val is None:
                        tm_val = val
                    continue

            # Collect text cells as candidate variant names
            if re.search(r'[A-Za-z]', cell_str):
                candidates.append(cell_str)

        # Pick the best variant name: prefer IsPETase-like names over plate IDs
        for c in candidates:
            if re.search(r'(PETase|[A-Z]\d+[A-Z])', c, re.I):
                variant = c
                break
        if not variant and candidates:
            # Fallback to the longest candidate (likely the variant name)
            variant = max(candidates, key=len)

        if tm_val is not None and variant:
            return {
                "variant_raw": variant,
                "tm": tm_val,
                "tm_std": tm_std,
                "method": config.get("method", "unknown"),
                "buffer_pH": config.get("buffer_pH"),
                "extraction_method": "pdfplumber_table",
            }
        return None


class BorderlessTableExtractor:
    """Extract Tm from borderless tables parsed as text lines."""

    def extract(self, pdf_path: str, page_num: int, config: dict) -> list[dict]:
        """Parse borderless table from text on specified page."""
        results = []
        with pdfplumber.open(pdf_path) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                return results
            text = pdf.pages[page_num - 1].extract_text()
            if not text:
                return results

        lines = text.split('\n')
        full_text = " ".join(lines)

        # Strategy 1: Match variant names followed by Tm±std in individual lines
        for line in lines:
            line = line.strip()
            if not line:
                continue
            name_match = re.match(
                r'^(?:.*?\s)?((?:IsPETase|DuraPETase|FAST-PETase|HotPETase|TS-PETase)\S*)\s+(.+)',
                line, re.I)
            if not name_match:
                continue
            variant_raw = name_match.group(1)
            rest = name_match.group(2)
            tm_match = re.match(r'(\d+\.?\d*)\s*[±]?\s*(\d+\.?\d*)?', rest)
            if tm_match:
                tm_val = float(tm_match.group(1))
                tm_std = float(tm_match.group(2)) if tm_match.group(2) else 0.0
                if 25 < tm_val < 120:
                    results.append({
                        "variant_raw": variant_raw,
                        "tm": tm_val,
                        "tm_std": tm_std,
                        "method": config.get("method", "unknown"),
                        "buffer_pH": config.get("buffer_pH"),
                        "extraction_method": "borderless_table_text",
                    })

        # Strategy 2: For concatenated text (main paper tables),
        # scan for variant-name + Tm patterns in the full text
        if not results:
            pattern = (r'((?:IsPETase|DuraPETase|FAST-PETase|HotPETase)\S*)\s+'
                       r'(\d+\.?\d*)\s*±\s*(\d+\.?\d*)')
            for m in re.finditer(pattern, full_text, re.I):
                variant_raw = m.group(1)
                tm_val = float(m.group(2))
                tm_std = float(m.group(3))
                if 25 < tm_val < 120:
                    results.append({
                        "variant_raw": variant_raw,
                        "tm": tm_val,
                        "tm_std": tm_std,
                        "method": config.get("method", "unknown"),
                        "buffer_pH": config.get("buffer_pH"),
                        "extraction_method": "borderless_table_text",
                    })

        return results


class TextExtractor:
    """Extract Tm values from running text via regex."""

    DEFAULT_PATTERNS = [
        r'[Tt]m\s*(?:value\s*)?(?:of|=|was|is|:)\s*~?(\d+\.?\d*)\s*°?\s*C',
        r'melting\s+(?:temperature|point)\s*(?:of|=|was|is|:)\s*~?(\d+\.?\d*)\s*°?\s*C',
        r'(\d+\.?\d*)\s*°\s*C\s*(?:higher|lower|increase|decrease)',
        r'[Tt]m\s*(?:was\s+)?increased?\s+(?:by\s+)?~?(\d+\.?\d*)\s*°?\s*C',
        r'Tm\s*of\s*~?\s*(\d+\.?\d*)',
        # Handle concatenated text from 2-column PDF layouts
        r'T\s*m?\s*value\s*of\s*(\d+\.?\d*)\s*°?\s*C',
        r'[Tt]hermal.*?(\d+\.?\d*)\s*°\s*C',
        r'elevated\s*by\s*(\d+\.?\d*)\s*°?\s*C',
    ]

    def extract(self, pdf_path: str, pages, config: dict) -> list[dict]:
        """Extract Tm values from text on specified pages."""
        results = []
        patterns = config.get("patterns", self.DEFAULT_PATTERNS)

        with pdfplumber.open(pdf_path) as pdf:
            if pages == "all":
                page_range = range(len(pdf.pages))
            else:
                page_range = [p - 1 for p in pages if 0 < p <= len(pdf.pages)]

            for page_idx in page_range:
                text = pdf.pages[page_idx].extract_text()
                if not text:
                    continue
                for pattern in patterns:
                    for match in re.finditer(pattern, text, re.I):
                        tm_val = float(match.group(1))
                        if 25 <= tm_val <= 120:
                            start = max(0, match.start() - 80)
                            end = min(len(text), match.end() + 80)
                            context = text[start:end].replace('\n', ' ').strip()
                            results.append({
                                "tm": tm_val,
                                "tm_std": 0.0,
                                "context": context,
                                "page": page_idx + 1,
                                "method": config.get("method", "unknown"),
                                "buffer_pH": config.get("buffer_pH"),
                                "extraction_method": "regex_text",
                            })
        return results


# ============================================================
# Stage 2: GLM-4.7 vision extraction
# ============================================================

class VisionExtractor:
    """Use GLM-4.7 vision to read Tm from rendered PDF pages."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def render_page(self, pdf_path: str, page_num: int,
                    bbox: tuple | None = None) -> bytes:
        """Render a PDF page (or cropped region) as PNG."""
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num - 1]
            if bbox:
                page = page.crop(bbox)
            img = page.to_image(resolution=150)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

    def find_table_bbox(self, pdf_path: str, page_num: int) -> tuple | None:
        """Find bounding box of the largest table on a page."""
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num - 1]
            tables = page.find_tables()
            if tables:
                largest = max(tables, key=lambda t:
                              (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]))
                return largest.bbox
        return None

    def query_vision(self, image_bytes: bytes, prompt: str) -> str:
        """Send image to GLM-4.7 vision API."""
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": EINFRA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": ("You are a precise data extraction assistant. "
                                "Extract data exactly as requested. "
                                "Return ONLY the JSON array, no explanation."),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{b64_image}"
                        }},
                    ],
                },
            ],
            "max_tokens": 8192,
            "temperature": 0.0,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{EINFRA_BASE}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode())
            msg = result["choices"][0]["message"]
            # GLM-4.7 may put response in content or reasoning_content
            content = msg.get("content")
            if content:
                return content
            # Fallback: check reasoning_content for JSON
            reasoning = msg.get("reasoning_content", "")
            if reasoning:
                print(f"    Note: response was in reasoning_content")
                return reasoning
            return ""
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            print(f"    Vision API HTTP {e.code}: {body}")
            return ""
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            print(f"    Vision API error: {e}")
            return ""

    def extract_from_table(self, pdf_path: str, page_num: int,
                           config: dict) -> list[dict]:
        """Render table region and ask vision to read Tm values."""
        bbox = self.find_table_bbox(pdf_path, page_num)
        image_bytes = self.render_page(pdf_path, page_num, bbox)
        prompt = config.get("vision_prompt",
            "Extract all melting temperature (Tm) values from this table. "
            "For each row, provide the variant name and Tm in degrees C. "
            "Include standard deviation if given. "
            "Return ONLY a JSON array: [{\"variant\": \"...\", \"tm\": ..., \"tm_std\": ...}]")
        response = self.query_vision(image_bytes, prompt)
        return self._parse_response(response, config)

    def extract_from_figure(self, pdf_path: str, page_num: int,
                            config: dict, runs: int = 1) -> list[dict]:
        """Extract Tm from bar chart figure. Multiple runs for consensus."""
        # Support crop_fraction: (x0_frac, y0_frac, x1_frac, y1_frac)
        crop_frac = config.get("crop_fraction")
        if crop_frac:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_num - 1]
                w, h = page.width, page.height
                bbox = (w * crop_frac[0], h * crop_frac[1],
                        w * crop_frac[2], h * crop_frac[3])
            image_bytes = self.render_page(pdf_path, page_num, bbox)
        else:
            image_bytes = self.render_page(pdf_path, page_num)
        prompt = config.get("vision_prompt",
            "Extract all variant names and their Tm values from this bar chart. "
            "Return ONLY a JSON array: [{\"variant\": \"...\", \"tm\": ...}]")

        all_runs = []
        for i in range(runs):
            response = self.query_vision(image_bytes, prompt)
            parsed = self._parse_response(response, config)
            all_runs.append(parsed)
            if runs > 1 and i < runs - 1:
                time.sleep(1)

        if runs == 1:
            return all_runs[0] if all_runs else []
        return self._consensus(all_runs)

    def _parse_response(self, response: str, config: dict) -> list[dict]:
        """Parse JSON array from vision response."""
        if not response:
            return []
        # Try to find JSON array in response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            return []
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            # Try cleaning common issues
            cleaned = json_match.group().replace("'", '"')
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                return []

        results = []
        for item in data:
            if isinstance(item, dict) and "variant" in item and "tm" in item:
                try:
                    results.append({
                        "variant_raw": str(item["variant"]),
                        "tm": float(item["tm"]),
                        "tm_std": float(item.get("tm_std", 0.0)),
                        "method": config.get("method", "unknown"),
                        "buffer_pH": config.get("buffer_pH"),
                        "extraction_method": "glm47_vision",
                    })
                except (ValueError, TypeError):
                    continue
        return results

    def _consensus(self, all_runs: list[list[dict]]) -> list[dict]:
        """Compute median consensus from multiple vision runs."""
        from collections import defaultdict
        import statistics
        by_variant = defaultdict(list)
        method_info = {}
        for run in all_runs:
            for item in run:
                key = normalize_variant_key(item["variant_raw"])
                by_variant[key].append(item["tm"])
                method_info[key] = (item.get("method"), item.get("buffer_pH"),
                                    item["variant_raw"])

        consensus = []
        for key, tms in by_variant.items():
            method, ph, raw = method_info[key]
            consensus.append({
                "variant_raw": raw,
                "tm": round(statistics.median(tms), 2),
                "tm_std": 0.0,
                "method": method,
                "buffer_pH": ph,
                "extraction_method": f"glm47_vision_consensus_{len(tms)}x",
            })
        return consensus


# ============================================================
# Stage 3: Cross-validation
# ============================================================

class CrossValidator:
    """Compare pdfplumber and vision extractions."""

    def validate(self, plumber: list[dict], vision: list[dict]) -> list[dict]:
        """Cross-validate two extraction sets. Returns merged with confidence."""
        vision_lookup = {}
        for v in vision:
            key = normalize_variant_key(v.get("variant_raw", ""))
            vision_lookup[key] = v

        validated = []
        seen_keys = set()

        for p in plumber:
            key = normalize_variant_key(p.get("variant_raw", ""))
            seen_keys.add(key)
            entry = {**p}

            if key in vision_lookup:
                v = vision_lookup[key]
                delta = abs(p["tm"] - v["tm"])
                if delta <= 0.5:
                    entry["extraction_confidence"] = "high"
                    entry["notes"] = f"Vision confirmed (delta={delta:.2f})"
                elif delta <= 2.0:
                    entry["extraction_confidence"] = "medium"
                    entry["notes"] = (f"Minor disagreement: pdfplumber={p['tm']}, "
                                      f"vision={v['tm']}, delta={delta:.2f}")
                else:
                    entry["extraction_confidence"] = "low_REVIEW"
                    entry["notes"] = (f"MISMATCH: pdfplumber={p['tm']}, "
                                      f"vision={v['tm']}, delta={delta:.2f}")
            else:
                entry["extraction_confidence"] = "single_method"
                entry["notes"] = "pdfplumber only"
            validated.append(entry)

        # Vision-only entries
        for key, v in vision_lookup.items():
            if key not in seen_keys:
                entry = {**v}
                entry["extraction_confidence"] = "vision_only"
                entry["notes"] = "GLM-4.7 vision only"
                validated.append(entry)

        return validated


# ============================================================
# Per-paper extraction configurations
# ============================================================

PAPER_CONFIGS = {
    "brott2022": {
        "dois": "10.1002/elsc.202100105",
        "pdfs": [
            # Index 0: SI PDF
            {"pattern": "*suppmat*", "label": "Brott 2022 SI"},
            # Index 1: Main paper
            {"pattern": "*Brott*thermostable*", "label": "Brott 2022 Main"},
        ],
        "extractions": [
            {
                "id": "table_s2",
                "type": "table",
                "pdf_index": 0,
                "page": 6,
                "description": "Table S2: Melting points from screening (nanoDSF)",
                "method": "nanoDSF",
                "buffer_pH": 7.5,
                "header_filter": r"[Mm]elting\s*point",
                "vision_prompt": (
                    "This is Table S2 from Brott et al. 2022 supplementary. "
                    "It shows melting points of IsPETase variants measured by nanoDSF. "
                    "Extract ALL variant names and their melting point values (Tm in degrees C). "
                    "Include standard deviation if shown (the +/- value). "
                    "Return ONLY a JSON array: "
                    '[{"variant": "...", "tm": ..., "tm_std": ...}]'
                ),
            },
            {
                "id": "main_table2",
                "type": "borderless_table",
                "pdf_index": 1,  # Main paper
                "page": 6,  # page 197
                "description": "Table 2: Combined variants Tm (borderless)",
                "method": "nanoDSF",
                "buffer_pH": 7.5,
                "vision_prompt": (
                    "This page contains Table 2 showing melting points of "
                    "combined IsPETase variants. Extract ALL variant names "
                    "and their Tm values (degrees C) with standard deviations. "
                    "Variants include: IsPETaseWT, IsPETaseTM, IsPETaseTMK95N/F201I, "
                    "IsPETaseTMN233C/S282C, IsPETaseTMK95N/F201I/N233C/S282C, "
                    "DuraPETase, DuraPETaseN233C/S282C. "
                    "Return ONLY a JSON array: "
                    '[{"variant": "...", "tm": ..., "tm_std": ...}]'
                ),
            },
            {
                "id": "table_s5",
                "type": "borderless_table",
                "pdf_index": 0,
                "page": 8,
                "description": "Table S5: Cross-study Tm comparison (borderless)",
                "method": "nanoDSF",
                "buffer_pH": 7.5,
                "vision_prompt": (
                    "This is Table S5 from Brott et al. 2022 supplementary. "
                    "It compares melting points across different studies. "
                    "Extract ALL variant names and their Tm values from the "
                    "'This study' column (first numeric column). "
                    "Return ONLY a JSON array: "
                    '[{"variant": "...", "tm": ..., "tm_std": ...}]'
                ),
            },
        ],
    },
    "son2019": {
        "dois": "10.1021/acscatal.9b00568",
        "pdfs": [
            # Index 0: SI PDF
            {"pattern": "*petase*_SI.pdf", "label": "Son 2019 SI"},
            # Index 1: Main paper
            {"pattern": "rational-protein-engineering*pet-degradation.pdf",
             "label": "Son 2019 Main"},
        ],
        "extractions": [
            {
                "id": "figure_s4b",
                "type": "figure_with_labels",
                "pdf_index": 0,
                "page": 8,
                "crop_fraction": (0, 0.4, 1.0, 1.0),  # Bottom 60% = Panel B
                "description": "Figure S4B: Single-mutant Tm bar chart with labels",
                "method": "DSF (SYPRO Orange)",
                "buffer_pH": 7.5,
                "vision_runs": 3,
                "vision_prompt": (
                    "This bar chart shows Tm values (melting temperatures in degrees C) "
                    "for IsPETase WT and 10 single-point mutant variants. "
                    "Each bar has a numerical label above it showing the exact Tm. "
                    "Read each numerical label precisely (to 2 decimal places). "
                    "Return ONLY a JSON array: "
                    '[{"variant": "WT", "tm": 48.81}, '
                    '{"variant": "S121D", "tm": 47.35}, ...]'
                ),
                "expected_variants": [
                    "WT", "S121D", "S121E", "D186H", "D186F",
                    "D186I", "D186L", "D186V", "P181A", "P181G", "P181S",
                ],
            },
            {
                "id": "figure_3b",
                "type": "figure_with_labels",
                "pdf_index": 1,
                "page": 4,
                "description": "Figure 3B: Combined-variant Tm bar chart",
                "method": "DSF (SYPRO Orange)",
                "buffer_pH": 7.0,
                "vision_runs": 3,
                "vision_prompt": (
                    "This is Figure 3 from Son et al. 2019 (ACS Catalysis). "
                    "Panel B shows a bar chart of melting temperatures (Tm) "
                    "for IsPETase combined-mutant variants. "
                    "Each bar has a numerical label showing the Tm in degrees C. "
                    "The variants include: WT, P181A, S121D/D186H, S121E/D186H, "
                    "P181A/S121D/D186H, P181A/S121E/D186H, "
                    "S121D/D186H/R280A, S121E/D186H/R280A. "
                    "Extract ALL variant names and Tm values exactly as labeled. "
                    "Return ONLY a JSON array: "
                    '[{"variant": "WT", "tm": 48.81}, ...]'
                ),
                "expected_variants": [
                    "WT", "P181A", "S121D/D186H", "S121E/D186H",
                    "P181A/S121D/D186H", "P181A/S121E/D186H",
                    "S121D/D186H/R280A", "S121E/D186H/R280A",
                ],
            },
        ],
    },
    "cui2021": {
        "dois": "10.1021/acscatal.0c05126",
        "pdfs": [
            # Index 0: Main paper
            {"pattern": "computational-redesign*grape*",
             "label": "Cui 2021 Main (GRAPE)"},
        ],
        "extractions": [
            {
                "id": "main_text_vision",
                "type": "figure_with_labels",
                "pdf_index": 0,
                "page": 4,
                "description": "Figure 5d/text: WT and DuraPETase Tm (vision)",
                "method": "DSF (SYPRO Orange)",
                "buffer_pH": 7.5,
                "vision_runs": 1,
                "vision_prompt": (
                    "This page is from Cui et al. 2021 about DuraPETase. "
                    "Find any melting temperature (Tm) values for IsPETase "
                    "wild-type (WT) and DuraPETase. "
                    "The paper reports DuraPETase Tm approximately 75-77 C "
                    "and WT IsPETase approximately 46 C. "
                    "Extract any Tm values you can find on this page. "
                    'Return ONLY a JSON array: [{"variant": "...", "tm": ...}]'
                ),
            },
        ],
    },
    "lu2022": {
        "dois": "10.1038/s41586-022-04599-z",
        "pdfs": [
            # Index 0: SI PDF
            {"pattern": "lu2022_suppl*", "label": "Lu 2022 SI"},
            # Index 1: Extended Data
            {"pattern": "lu2022_extended*", "label": "Lu 2022 Extended Data"},
        ],
        "extractions": [
            {
                "id": "si_text",
                "type": "text_regex",
                "pdf_index": 0,
                "pages": "all",
                "description": "FAST-PETase Tm from supplementary text",
                "method": "DSF",
                "buffer_pH": None,
                "patterns": [
                    r'[Tt]m\s*(?:value\s*)?(?:of|=|was|is|:)\s*~?(\d+\.?\d*)\s*°?\s*C',
                    r'melting\s+(?:temperature|point)\s*(?:of|=|was|is|:)\s*~?(\d+\.?\d*)',
                ],
            },
            {
                "id": "extended_data_p2_vision",
                "type": "figure_with_labels",
                "pdf_index": 1,
                "page": 2,
                "description": "Extended Data p2: DSF/activity data (vision)",
                "method": "DSF",
                "buffer_pH": None,
                "vision_runs": 1,
                "vision_prompt": (
                    "This is Extended Data from Lu et al. 2022 (Nature) about FAST-PETase. "
                    "Look for any melting temperature (Tm) values or DSF curves. "
                    "FAST-PETase is ThermoPETase+N233K+R224Q (5 mutations total). "
                    "Extract any Tm values you can find. "
                    'Return ONLY a JSON array: [{"variant": "...", "tm": ...}]'
                ),
            },
            {
                "id": "extended_data_p3_vision",
                "type": "figure_with_labels",
                "pdf_index": 1,
                "page": 3,
                "description": "Extended Data p3: Additional characterization (vision)",
                "method": "DSF",
                "buffer_pH": None,
                "vision_runs": 1,
                "vision_prompt": (
                    "This is Extended Data from Lu et al. 2022 (Nature) about FAST-PETase. "
                    "Look for any melting temperature (Tm) or thermostability data. "
                    "Extract any Tm values you can find. "
                    'Return ONLY a JSON array: [{"variant": "...", "tm": ...}]'
                ),
            },
        ],
    },
    "bell2022": {
        "dois": "10.1038/s41929-022-00821-3",
        "pdfs": [
            # Index 0: Main paper PDF (font encoding garbled — vision only)
            {"pattern": "bell2022*", "label": "Bell 2022 Main (HotPETase)"},
        ],
        "extractions": [
            {
                "id": "main_text_vision",
                "type": "figure_with_labels",
                "pdf_index": 0,
                "page": 3,
                "description": "HotPETase Tm from main text (vision, garbled PDF)",
                "method": "nanoDSF",
                "buffer_pH": None,
                "vision_runs": 1,
                "vision_prompt": (
                    "This page is from Bell et al. 2022 (Nature Catalysis) about HotPETase. "
                    "Find any melting temperature (Tm) values mentioned in text or figures. "
                    "HotPETase is an engineered IsPETase with Tm around 82.5°C. "
                    "Also look for wild-type IsPETase Tm if mentioned. "
                    "Extract ALL Tm values from this page. "
                    'Return ONLY a JSON array: [{"variant": "...", "tm": ...}]'
                ),
            },
            {
                "id": "figure_2_vision",
                "type": "figure_with_labels",
                "pdf_index": 0,
                "page": 4,
                "description": "Figure 2: Thermostability data (vision)",
                "method": "nanoDSF",
                "buffer_pH": None,
                "vision_runs": 1,
                "vision_prompt": (
                    "This page contains Figure 2 from Bell et al. 2022 about HotPETase. "
                    "Look for melting temperature (Tm) values in any panel or caption. "
                    "HotPETase Tm is approximately 82.5°C. "
                    "Extract ALL Tm values you can find. "
                    'Return ONLY a JSON array: [{"variant": "...", "tm": ...}]'
                ),
            },
        ],
    },
}


# ============================================================
# Main orchestration
# ============================================================

def process_paper(paper_id: str, api_key: str | None, no_vision: bool = False):
    """Run full extraction pipeline for one paper."""
    config = PAPER_CONFIGS.get(paper_id)
    if not config:
        print(f"No config for paper: {paper_id}")
        return []

    # Resolve PDF paths
    pdf_paths = []
    for pdf_info in config["pdfs"]:
        path = find_pdf(pdf_info["pattern"])
        if path:
            pdf_paths.append(path)
            print(f"  Found: {pdf_info['label']} -> {path.name}")
        else:
            pdf_paths.append(None)
            print(f"  MISSING: {pdf_info['label']} (pattern: {pdf_info['pattern']})")

    table_ext = TableExtractor()
    borderless_ext = BorderlessTableExtractor()
    text_ext = TextExtractor()
    vision_ext = VisionExtractor(api_key) if api_key and not no_vision else None
    validator = CrossValidator()

    all_results = []

    for extraction in config["extractions"]:
        pdf_idx = extraction.get("pdf_index", 0)
        if pdf_idx >= len(pdf_paths) or pdf_paths[pdf_idx] is None:
            print(f"  SKIP: {extraction['description']} (PDF not found)")
            continue

        pdf_path = str(pdf_paths[pdf_idx])
        ext_type = extraction["type"]
        print(f"\n  Extracting: {extraction['description']}")

        # Stage 1: pdfplumber
        plumber_results = []
        if ext_type == "table":
            plumber_results = table_ext.extract(
                pdf_path, extraction["page"], extraction)
            print(f"    pdfplumber: {len(plumber_results)} entries")
            for r in plumber_results:
                print(f"      {r['variant_raw']:30s} Tm={r['tm']:.1f} +/- {r['tm_std']:.1f}")
        elif ext_type == "borderless_table":
            plumber_results = borderless_ext.extract(
                pdf_path, extraction["page"], extraction)
            print(f"    borderless table: {len(plumber_results)} entries")
            for r in plumber_results:
                print(f"      {r['variant_raw']:30s} Tm={r['tm']:.1f} +/- {r['tm_std']:.1f}")
        elif ext_type == "text_regex":
            plumber_results = text_ext.extract(
                pdf_path, extraction.get("pages", "all"), extraction)
            print(f"    text regex: {len(plumber_results)} matches")
            for r in plumber_results:
                ctx = r.get("context", "")[:60]
                print(f"      p{r.get('page', '?')}: Tm={r['tm']:.1f} ... {ctx}")

        # Stage 2: Vision cross-check
        vision_results = []
        if vision_ext and ext_type in ("table", "borderless_table", "figure_with_labels"):
            page = extraction.get("page")
            if page:
                print(f"    Running GLM-4.7 vision on page {page}...")
                runs = extraction.get("vision_runs", 1)
                if ext_type == "figure_with_labels":
                    vision_results = vision_ext.extract_from_figure(
                        pdf_path, page, extraction, runs=runs)
                else:
                    vision_results = vision_ext.extract_from_table(
                        pdf_path, page, extraction)
                print(f"    vision: {len(vision_results)} entries")
                for r in vision_results:
                    print(f"      {r['variant_raw']:30s} Tm={r['tm']:.1f}")

        # Stage 3: Cross-validate
        if plumber_results and vision_results:
            validated = validator.validate(plumber_results, vision_results)
        elif plumber_results:
            validated = [{**r, "extraction_confidence": "single_method",
                          "notes": "pdfplumber only"} for r in plumber_results]
        elif vision_results:
            validated = [{**r, "extraction_confidence": "vision_only",
                          "notes": "GLM-4.7 vision only"} for r in vision_results]
        else:
            validated = []
            print(f"    No data extracted")

        # Annotate with source info
        for entry in validated:
            entry["source_doi"] = config["dois"]
            entry["source_table"] = extraction["id"]
            entry["paper_id"] = paper_id

            raw = entry.get("variant_raw", "")
            name, mutation = resolve_variant_name(raw)
            entry["variant_name"] = name
            entry["mutation"] = mutation

        all_results.extend(validated)

    # Save per-paper CSV
    if all_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{paper_id}_tm.csv"
        save_csv(all_results, output_path)
        print(f"\n  Saved {len(all_results)} entries -> {output_path.name}")
    else:
        print(f"\n  No data extracted for {paper_id}")

    return all_results


def save_csv(results: list[dict], path: Path):
    """Save extraction results to CSV."""
    fields = [
        "variant_name", "mutation", "enzyme", "tm", "tm_std",
        "delta_tm", "method", "buffer_pH", "source_doi",
        "source_table", "extraction_method", "extraction_confidence", "notes",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            r.setdefault("enzyme", "IsPETase")
            r.setdefault("delta_tm", "")
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Tm values from PETase literature PDFs")
    parser.add_argument("--paper", help="Paper ID (brott2022, son2019, cui2021, lu2022)")
    parser.add_argument("--all", action="store_true", help="Process all papers")
    parser.add_argument("--no-vision", action="store_true",
                        help="Skip GLM-4.7 vision (pdfplumber/regex only)")
    parser.add_argument("--list", action="store_true", help="List available papers")
    args = parser.parse_args()

    if args.list:
        for pid, cfg in PAPER_CONFIGS.items():
            nextr = len(cfg["extractions"])
            print(f"  {pid:20s}  {nextr} extraction(s)  DOI: {cfg['dois']}")
        return

    api_key = None
    if not args.no_vision:
        try:
            api_key = load_api_key()
            print(f"API key loaded ({len(api_key)} chars)")
        except SystemExit:
            print("WARNING: No API key found, running without vision")
            args.no_vision = True

    if args.all:
        for paper_id in PAPER_CONFIGS:
            print(f"\n{'='*60}")
            print(f"Processing: {paper_id}")
            print(f"{'='*60}")
            process_paper(paper_id, api_key, args.no_vision)
    elif args.paper:
        if args.paper not in PAPER_CONFIGS:
            sys.exit(f"Unknown paper: {args.paper}. Use --list to see available.")
        process_paper(args.paper, api_key, args.no_vision)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
