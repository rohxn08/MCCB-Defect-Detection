"""
JSON Formatter — Formatting functions for MCCB OCR pipeline output.

Provides functions to normalize, pivot, and clean OCR-extracted table data
into structured JSON format. Imported by robust_ocr_pipeline.py.
"""

import json
import re
import os



# ─────────────────────────────────────────────
#  FORMATTING HELPERS  (ported from prototype)
# ─────────────────────────────────────────────

def build_table_columns(rows):
    """
    Converts vertical table rows (first col = header, rest = values)
    into a column-keyed dict.

    Rules:
    - If a field has only one value → store as string
    - If a field has multiple values → store as list
    - Rows with no values after the header → skipped
    """
    result = {}

    for row in rows:
        if not row:
            continue

        header = row[0].strip()
        values = [v.strip() for v in row[1:] if v.strip()]  # drop blank cells

        if not values:
            continue

        if len(values) == 1:
            result[header] = values[0]   # Single value → plain string
        else:
            result[header] = values      # Multiple values → list

    return result


def filter_nulls(data):
    """
    Recursively removes keys with null-like values:
    None, "", "Not Found", [], {}
    """
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            v = filter_nulls(v)
            if v in (None, "", "Not Found", [], {}):
                continue
            cleaned[k] = v
        return cleaned

    if isinstance(data, list):
        cleaned = [filter_nulls(i) for i in data]
        cleaned = [i for i in cleaned if i not in (None, "", "Not Found", [], {})]
        return cleaned

    return data


def extract_top_section_fields(top_lines):
    """
    Scans raw OCR lines from the top of the label to pull out:
    - product_name: e.g. "SACE FORMULA P1N 160"
    - serial_number: e.g. "CFC6306263"  (from lines containing S/N)

    Strategy:
    - S/N line is matched via regex → serial_number
    - The longest remaining line that looks like a product name is used
      (excludes single words like brand names, and the S/N line itself)
    """
    sn_pattern = re.compile(r'S/?N\s*[:\-]?\s*([A-Z0-9]+)', re.IGNORECASE)

    serial_number = None
    product_name  = None
    sn_line       = None

    # Pass 1 — find S/N
    for line in top_lines:
        m = sn_pattern.search(line)
        if m:
            serial_number = m.group(1).strip()
            sn_line = line
            break

    # Pass 2 — find product name (longest multi-word line that isn't the S/N line)
    candidates = [
        line for line in top_lines
        if line != sn_line and len(line.split()) >= 2
    ]
    if candidates:
        product_name = max(candidates, key=len)  # longest = most descriptive

    return product_name, serial_number


# Known electrical-parameter headers (used for orientation detection + filtering)
_TABLE_HEADER_RE = re.compile(r'^(Ue|Icu|Ics|Uimp|Ui)\b', re.IGNORECASE)


def _is_table_header(text):
    """Return True if *text* starts with a known table field name."""
    return bool(_TABLE_HEADER_RE.match(text.strip()))


def _transpose_rows(rows):
    """Transpose a ragged list-of-lists (pad shorter rows with '')."""
    if not rows:
        return rows
    max_len = max(len(r) for r in rows)
    padded = [r + [''] * (max_len - len(r)) for r in rows]
    return [list(col) for col in zip(*padded)]


def normalize_table(rows):
    """
    Detect table orientation and normalize so that each row starts
    with a known header (Ue, Icu, Ics, Ui, Uimp) followed by its values.

    P-series labels have headers across the FIRST ROW  → needs transpose.
    XT-series labels have headers down the FIRST COLUMN → already correct.

    After orientation fix, rows that don't start with a known header
    are discarded (removes noise like CAT.A, ABB, IS/IEC, etc.).
    """
    if not rows or len(rows) < 2:
        return rows

    # Count known headers in the first row vs the first column
    first_row_headers = sum(1 for cell in rows[0] if _is_table_header(cell))
    first_col_headers = sum(1 for r in rows if r and _is_table_header(r[0]))

    # If more headers sit across the first row → table is rotated
    if first_row_headers > first_col_headers:
        rows = _transpose_rows(rows)

    # Keep only rows whose first cell is a known header
    filtered = []
    for row in rows:
        if row and _is_table_header(row[0]):
            # Drop empty / blank trailing cells
            clean = [row[0]] + [v for v in row[1:] if v.strip()]
            filtered.append(clean)

    # --- Trim noise values from the end of each row ---
    # Determine the "true" data width.
    # Primary: count voltage-pattern values in the Ue row. Voltage values
    # look like "220-240Vac", "500Vac", "230/240", "690", "250Vdc" etc.
    # Noise like "Tested at415Vac", "ABB", "IS/IEC 60947-2" is excluded.
    # Fallback: count ALL numeric values in Icu/Ics rows.
    data_width = None
    voltage_re = re.compile(r'^\d+[/\-]?\d*\s*(Vac|Vdc|V)?$', re.IGNORECASE)

    # Try Ue row first
    for row in filtered:
        header_lower = row[0].strip().lower()
        if header_lower.startswith('ue'):
            data_width = sum(1 for v in row[1:] if voltage_re.match(v.strip()))
            break

    # Fallback to Icu/Ics — count total numeric values
    if data_width is None:
        for row in filtered:
            header_lower = row[0].strip().lower()
            if header_lower.startswith(('icu', 'ics')):
                count = sum(1 for v in row[1:] if re.match(r'^[\d.]+$', v.strip()))
                if data_width is None or count < data_width:
                    data_width = count

    # Truncate every row to header + data_width values
    if data_width and data_width > 0:
        trimmed = []
        for row in filtered:
            header_lower = row[0].strip().lower()
            if header_lower.startswith(('ui',)) and not header_lower.startswith(('uimp',)):
                # Ui has exactly 1 value
                trimmed.append(row[:2])
            elif header_lower.startswith('uimp'):
                # Uimp has exactly 1 value
                trimmed.append(row[:2])
            else:
                trimmed.append(row[:1 + data_width])
        filtered = trimmed

    return filtered


def extract_embedded_fields(table_rows):
    """
    Scan raw table rows for product info that may be embedded in a
    non-header row.  Common on XT-series labels where the first row
    looks like: ["Tmax XT1S 160", "Ui=800V", "Uimp=8kV", "S/N:CFC6305916"]

    Returns dict with optional keys: product_name, serial_number, ui, uimp
    """
    fields = {}

    ui_re   = re.compile(r'U[ij]\s*=\s*(\d+\s*V)', re.IGNORECASE)
    uimp_re = re.compile(r'Uimp\s*=\s*(\d+\s*kV)', re.IGNORECASE)
    sn_re   = re.compile(r'S/?N\s*[:\-]?\s*([A-Z0-9]+)', re.IGNORECASE)

    for row in table_rows:
        # Skip rows that start with a known table header — those are real data
        if row and _is_table_header(row[0]):
            continue

        for cell in row:
            # Uimp (check before Ui since "Uimp" contains "Ui")
            m = uimp_re.search(cell)
            if m and 'uimp' not in fields:
                fields['uimp'] = m.group(1).strip()

            # Ui (only match if it's NOT a Uimp hit)
            if 'ui' not in fields:
                m = ui_re.search(cell)
                if m and not uimp_re.search(cell):
                    fields['ui'] = m.group(1).strip()
                elif m and uimp_re.search(cell):
                    # Cell has both patterns, e.g. "Tmax XT1C 160 Ui=800V"
                    # Only take if the Ui match position differs from the Uimp match
                    ui_match = ui_re.search(cell)
                    uimp_match = uimp_re.search(cell)
                    if ui_match and uimp_match and ui_match.start() != uimp_match.start():
                        fields['ui'] = ui_match.group(1).strip()

            # Serial number
            m = sn_re.search(cell)
            if m and 'serial_number' not in fields:
                fields['serial_number'] = m.group(1).strip()

        # Product name — first cell of a non-header row, multi-word, not noise
        if row and 'product_name' not in fields:
            candidate = row[0].strip()
            if not _is_table_header(candidate) and len(candidate.split()) >= 2:
                noise = ('is/iec', 'nema', 'cat.', 'hz', 'tested')
                if not any(n in candidate.lower() for n in noise):
                    # Strip embedded Ui=/Uimp= from the name
                    clean = ui_re.sub('', candidate)
                    clean = uimp_re.sub('', clean).strip()
                    if clean and len(clean.split()) >= 2:
                        fields['product_name'] = clean

    return fields


def format_image_result(raw_data):
    """
    Takes the raw dict from RobustMCCBPipeline.process_image() and
    builds the final formatted dict for one image, applying:
      - top-section field extraction (product_name, serial_number)
      - embedded field extraction (XT-series fallback)
      - table orientation fix + noise filtering
      - table pivoting (list-of-lists → keyed dict)
      - null filtering
    """
    entry = {}

    # --- Top-section fields ---
    top_lines = raw_data.get("top_lines", [])
    if top_lines:
        product_name, serial_number = extract_top_section_fields(top_lines)
        if product_name:
            entry["product_name"] = product_name
        if serial_number:
            entry["serial_number"] = serial_number

    # --- Table data ---
    table_rows = raw_data.get("table_data", [])

    if table_rows:
        # Scan for embedded fields (XT-series has product info in table rows)
        embedded = extract_embedded_fields(table_rows)

        # Fill gaps from embedded extraction
        if "product_name" not in entry and "product_name" in embedded:
            entry["product_name"] = embedded["product_name"]
        if "serial_number" not in entry and "serial_number" in embedded:
            entry["serial_number"] = embedded["serial_number"]

        # Normalize orientation + filter noise, then pivot
        normalized = normalize_table(table_rows)
        table_dict = build_table_columns(normalized)

        # Inject Ui / Uimp from embedded if not already in the table
        if "ui" in embedded:
            has_ui = any(
                k.lower().startswith('ui') and not k.lower().startswith('uimp')
                for k in table_dict
            )
            if not has_ui:
                table_dict["Ui"] = embedded["ui"]

        if "uimp" in embedded:
            has_uimp = any(k.lower().startswith('uimp') for k in table_dict)
            if not has_uimp:
                table_dict["Uimp"] = embedded["uimp"]

        entry["table_data"] = table_dict

    # Rating
    rating = raw_data.get("rating")
    if rating:
        entry["rating"] = rating

    return filter_nulls(entry)
