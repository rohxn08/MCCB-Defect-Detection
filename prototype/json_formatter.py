import json
import re

# ─────────────────────────────────────────────
#  MOCK OCR table_rows output (list of lists)
#  Replace these with actual pipeline output to test
# ─────────────────────────────────────────────

# P-Series (SACE FORMULA P1B 160) - Vertical table from real label
# First column = field name, remaining columns = values per voltage level
mock_p_series_rows = [
    ["Ue",        "220-240Vac", "380-415Vac", "440Vac", "500Vac", "250Vdc"],
    ["Icu (kA)",  "25",         "18",         "15",     "3",      "18"],
    ["Ics (kA)",  "25",         "18",         "7.5",    "3",      "18"],
    ["Ui",        "750V"],               # Single value
    ["Uimp",      "8kV"],                # Single value
]

# XT-Series (Tmax XT1S 160) - Vertical table from real label
mock_xt_series_rows = [
    ["Ue (V)",      "230/240", "415", "440", "525", "690"],
    ["Icu (kA)",    "85",      "50",  "50",  "35",  "8"],
    ["Ics (%Icu)",  "95",      "75",  "50",  "50",  "50"],
    ["Ui",          "800V"],
    ["Uimp",        "8kV"],
]

# Rating detected from the label's rating ROI (below the switch)
mock_p_rating  = "In=100A"
mock_xt_rating = "In=40A"

# Mock top-section OCR lines (raw text lines from the top ~10% of the image)
# In real pipeline these come from all_lines where cy < table_y_min
mock_p_top_lines = [
    "ABB",
    "SACE FORMULA P1B 160",
    "S/N CFC6306263",
]

mock_xt_top_lines = [
    "ABB",
    "Tmax XT1S 160",
    "S/N CFC6305916",
]


# ─────────────────────────────────────────────
#  HELPERS
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
    - serial_number: e.g. "1SVR040012R1000"  (from lines containing S/N)

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


def format_image_result(filename, table_rows, top_lines=None, rating=None):
    """
    Builds the final dict for one image, applying table pivoting + null filtering.
    """
    entry = {}

    # Top-section fields
    if top_lines:
        product_name, serial_number = extract_top_section_fields(top_lines)
        if product_name:
            entry["product_name"] = product_name
        if serial_number:
            entry["serial_number"] = serial_number

    if table_rows:
        entry["table_data"] = build_table_columns(table_rows)

    if rating:
        entry["rating"] = rating

    return filter_nulls(entry)


# ─────────────────────────────────────────────
#  MAIN — Simulate processing two images
# ─────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    # Simulate P-Series image
    results["master_P1B.jpg"] = format_image_result(
        filename="master_P1B.jpg",
        top_lines=mock_p_top_lines,
        table_rows=mock_p_series_rows,
        rating=mock_p_rating
    )

    # Simulate XT-Series image
    results["master_XT1S.jpg"] = format_image_result(
        filename="master_XT1S.jpg",
        top_lines=mock_xt_top_lines,
        table_rows=mock_xt_series_rows,
        rating=mock_xt_rating  # "Not Found" → will be filtered out
    )

    # Pretty print to console
    print(json.dumps(results, indent=2))

    # Optionally save to file
    output_path = "prototype/results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
