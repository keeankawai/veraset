"""Builds "Task 1/WSLCB Dispensary Matching.xlsx": every approved link between a WSLCB storefront and a Veraset place label, side by side, so each
one can be checked by hand. Reads [1a], [1e], [6a] and veraset_poi_months from processed-data/ (run the step2_ and step6_ notebooks first).
Run from the project root: python3 tools/build_match_audit.py"""
import polars as pl
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

OUT = "Task 1/WSLCB Dispensary Matching.xlsx"
STOREFRONT = ["street_address_norm", "city_norm"]
LABEL = ["safegraph_place_id", "location_name", "street_address", "city"]

e = pl.read_parquet("processed-data/1e_storefront_place_months.parquet")
locations = pl.read_parquet("processed-data/1a_wslcb_locations.parquet")
coverage = pl.read_parquet("processed-data/6a_storefront_coverage.parquet")

# (1) WSLCB side, one row per storefront (normalized address + city, every license ever operating there)
def join_unique(col):
    return pl.col(col).drop_nulls().cast(pl.String).unique().sort().str.join("; ")

wslcb = (locations
    .filter(pl.col("ever_operating") & ~pl.col("is_placeholder_address"))
    .group_by(STOREFRONT)
    .agg(
        join_unique("street_address").alias("wslcb_address"),
        join_unique("zip5").alias("wslcb_zip"),
        join_unique("county").alias("wslcb_county"),
        pl.col("tradenames").list.explode().drop_nulls().unique().sort().str.join("; ").alias("wslcb_tradenames"),
        pl.col("first_operating").min().alias("wslcb_first_operating"),
        pl.col("last_operating").max().alias("wslcb_last_operating"),
    )
    .join(coverage.select(STOREFRONT + ["place_name", "place_type"]), on = STOREFRONT, how = "left")
)

# (2) Veraset side: the zip code each label is listed with (most common across months)
labels_used = e.select(LABEL).unique()
veraset_zip = (pl.scan_parquet("processed-data/veraset_poi_months.parquet")
    .join(labels_used.lazy(), on = LABEL, how = "semi", nulls_equal = True)
    .group_by(LABEL + ["zip_code"])
    .agg(pl.len().alias("n"))
    .sort("n", descending = True)
    .group_by(LABEL)
    .agg(pl.col("zip_code").first().alias("veraset_zip"))
    .collect(engine = "streaming")
)

# (3) One row per link (storefront x label), months collapsed
def collapse(rows):
    return (rows
        .group_by(STOREFRONT + LABEL)
        .agg(
            pl.col("placekey").drop_nulls().first(),
            pl.col("naics_code").drop_nulls().first(),
            # the checks as of the approved months (a few labels change NAICS code between months, which flips the name check in the others)
            pl.coalesce(pl.col("address_match").filter(pl.col("approved")).first(), pl.col("address_match").first()).alias("address_match"),
            pl.coalesce(pl.col("name_rule").filter(pl.col("approved")).first(), pl.col("name_rule").first()).alias("name_rule"),
            pl.coalesce(pl.col("name_check").filter(pl.col("approved")).first(), pl.col("name_check").filter(pl.col("name_check") != "no").first()).alias("name_check"),
            pl.col("name_overlap").max(),
            pl.col("outline_evidence").drop_nulls().unique().sort().str.join(" / ").alias("outline_evidence"),
            pl.col("month").filter(pl.col("approved")).len().alias("approved_months"),
            pl.col("month").filter(pl.col("approved")).min().alias("first_approved"),
            pl.col("month").filter(pl.col("approved")).max().alias("last_approved"),
            pl.col("month").len().alias("months_listed"),
            pl.col("month").min().alias("first_listed"),
            pl.col("month").max().alias("last_listed"),
            pl.col("num_visits").filter(pl.col("approved")).sum().alias("approved_visits"),
            pl.col("current_list_only").filter(pl.col("approved")).sum().alias("current_list_only_months"),
            pl.col("operating_licenses").filter(pl.col("approved")).list.explode().drop_nulls().unique().sort().cast(pl.String).str.join("; ").alias("licenses_in_months"),
            pl.col("approved").any(),
        )
        .join(wslcb, on = STOREFRONT, how = "left")
        .join(veraset_zip, on = LABEL, how = "left", nulls_equal = True)
        .sort(["wslcb_county", "city_norm", "street_address_norm", "approved_visits"], descending = [False, False, False, True])
    )

matches = collapse(e.filter(pl.col("approved").any().over(STOREFRONT + LABEL)))
storefront_numbers = (matches.select(STOREFRONT).unique(maintain_order = True)
    .with_row_index("storefront_number", offset = 1).with_columns(pl.col("storefront_number").cast(pl.Int64)))
matches = matches.join(storefront_numbers, on = STOREFRONT, how = "left", maintain_order = "left")

# Links that were considered but never approved: rejected by hand, or the name was fine but no license was operating in any month the label shows up
not_approved = collapse(e.filter(~pl.col("approved").any().over(STOREFRONT + LABEL) & (pl.col("name_check") != "no")))
not_approved = not_approved.with_columns(
    pl.when(pl.col("name_check") == "no_in_review").then(pl.lit("Rejected in manual review"))
      .otherwise(pl.lit("Name OK, but no WSLCB license operating there in any month the label appears"))
      .alias("why_not")
)

unmatched = (coverage
    .filter(~pl.col("matched"))
    .join(wslcb.select(STOREFRONT + ["wslcb_address", "wslcb_zip", "wslcb_county", "wslcb_tradenames", "wslcb_first_operating", "wslcb_last_operating"]),
          on = STOREFRONT, how = "left")
    .with_columns(pl.col("licenses").cast(pl.List(pl.String)).list.join("; ").alias("licenses"))
    .sort(["wslcb_county", "city_norm", "street_address_norm"])
)

ADDRESS_TEXT = {"exact": "Same address and city",
                "same_address_and_zip": "Same address and ZIP, city named differently",
                "different_address": "Different address (manual review)"}
NAME_TEXT = {"auto": "Automatic: name matches a WSLCB trade name",
             "yes_in_review": "Manual review: yes",
             "no_in_review": "Manual review: no",
             "review": "Not reviewed"}
OUTLINE_TEXT = {"own": "own", "likely_own": "likely own", "shared": "SHARED with another business",
                "likely_shared": "likely shared", "unknown": "unknown"}

def title(s):
    return s.title() if s else s

def month_range(first, last, n):
    return f"{first} to {last} ({n})" if first else ""

# (4) Workbook
FONT = "Arial"
thin = Side(style = "thin", color = "BFBFBF")
BORDER = Border(left = thin, right = thin, top = thin, bottom = thin)
FILLS = {
    "wslcb": PatternFill("solid", fgColor = "DCE6F1"),
    "veraset": PatternFill("solid", fgColor = "E2EFDA"),
    "why": PatternFill("solid", fgColor = "FFF2CC"),
    "visits": PatternFill("solid", fgColor = "EDEDED"),
    "audit": PatternFill("solid", fgColor = "FFFF00"),
    "id": PatternFill("solid", fgColor = "D9D9D9"),
}
BAND = PatternFill("solid", fgColor = "F7F7F7")
GOOD = PatternFill("solid", fgColor = "C6EFCE")
WARN = PatternFill("solid", fgColor = "FFEB9C")
BAD = PatternFill("solid", fgColor = "FFC7CE")

def write_table(ws, groups, rows, band_key = None, row_fills = None):
    """groups: [(group name, fill key, [(header, width, value function)])]. Two header rows, then data."""
    col = 1
    columns = []
    for group, fill_key, cols in groups:
        ws.cell(1, col, group)
        if len(cols) > 1:
            ws.merge_cells(start_row = 1, start_column = col, end_row = 1, end_column = col + len(cols) - 1)
        for i, (header, width, fn) in enumerate(cols):
            for r in (1, 2):
                cell = ws.cell(r, col + i)
                cell.fill = FILLS[fill_key]
                cell.font = Font(name = FONT, bold = True, size = 11 if r == 1 else 10)
                cell.alignment = Alignment(horizontal = "center" if r == 1 else "left", vertical = "center", wrap_text = True)
                cell.border = BORDER
            ws.cell(2, col + i, header)
            ws.column_dimensions[get_column_letter(col + i)].width = width
            columns.append((header, fn))
        col += len(cols)
    ws.row_dimensions[2].height = 45
    last_band, shade = None, False
    for r, row in enumerate(rows, start = 3):
        if band_key:
            if row[band_key] != last_band:
                shade, last_band = not shade, row[band_key]
        for c, (header, fn) in enumerate(columns, start = 1):
            cell = ws.cell(r, c, fn(row))
            cell.font = Font(name = FONT, size = 10)
            cell.alignment = Alignment(vertical = "top", wrap_text = True)
            cell.border = BORDER
            if shade:
                cell.fill = BAND
            if row_fills and header in row_fills:
                fill = row_fills[header](row)
                if fill:
                    cell.fill = fill
    ws.freeze_panes = ws.cell(3, 4)
    ws.auto_filter.ref = f"A2:{get_column_letter(len(columns))}{max(2 + len(rows), 3)}"
    return len(columns)

wb = Workbook()
readme = wb.active
readme.title = "Read me"

# Matches sheet
ws = wb.create_sheet("Matches")
match_rows = matches.to_dicts()
for i, row in enumerate(match_rows, start = 1):
    row["match_id"] = f"M{i:04d}"

groups = [
    ("", "id", [("Match ID", 9, lambda r: r["match_id"]),
                ("Storefront #", 10, lambda r: r["storefront_number"]),
                ("County", 12, lambda r: r["wslcb_county"])]),
    ("WSLCB record (the licensed store)", "wslcb", [
        ("Trade name(s)", 28, lambda r: r["wslcb_tradenames"]),
        ("Street address", 26, lambda r: r["wslcb_address"]),
        ("City", 14, lambda r: title(r["city_norm"])),
        ("ZIP", 8, lambda r: r["wslcb_zip"]),
        ("Licenses operating in the approved months", 14, lambda r: r["licenses_in_months"]),
        ("Operating per WSLCB (first to last)", 14, lambda r: f"{r['wslcb_first_operating']} to {r['wslcb_last_operating']}" if r["wslcb_first_operating"] else "current list only"),
        ("Census place of the store", 14, lambda r: r["place_name"] or "(outside any place)")]),
    ("Veraset / SafeGraph record (the place visits are attributed to)", "veraset", [
        ("Location name", 28, lambda r: r["location_name"]),
        ("Street address", 26, lambda r: r["street_address"]),
        ("City", 14, lambda r: r["city"]),
        ("ZIP", 8, lambda r: r["veraset_zip"]),
        ("NAICS", 9, lambda r: r["naics_code"]),
        ("SafeGraph place ID", 24, lambda r: r["safegraph_place_id"]),
        ("Placekey", 20, lambda r: r["placekey"])]),
    ("Why it was matched", "why", [
        ("Address check", 22, lambda r: ADDRESS_TEXT[r["address_match"]]),
        ("Name check", 22, lambda r: NAME_TEXT[r["name_check"]]),
        ("Name overlap with best trade name (0-1)", 11, lambda r: None if r["name_overlap"] is None else round(r["name_overlap"], 2)),
        ("Building outline (Oct 2026, flag only)", 16, lambda r: " / ".join(OUTLINE_TEXT[x] for x in r["outline_evidence"].split(" / ")) if r["outline_evidence"] else "")]),
    ("Visits counted", "visits", [
        ("Approved months (first to last, count)", 22, lambda r: month_range(r["first_approved"], r["last_approved"], r["approved_months"])),
        ("Months the label is listed at this storefront", 22, lambda r: month_range(r["first_listed"], r["last_listed"], r["months_listed"])),
        ("Approved months backed only by the current WSLCB list", 12, lambda r: r["current_list_only_months"]),
        ("Visits counted", 10, lambda r: r["approved_visits"])]),
    ("Your audit (fill in)", "audit", [
        ("Verdict", 12, lambda r: None),
        ("Notes", 36, lambda r: None)]),
]
row_fills = {
    "Address check": lambda r: {"exact": GOOD, "same_address_and_zip": WARN, "different_address": BAD}[r["address_match"]],
    "Name check": lambda r: GOOD if r["name_check"] == "auto" else WARN,
    "Building outline (Oct 2026, flag only)": lambda r: WARN if r["outline_evidence"] and "shared" in r["outline_evidence"] else None,
    "Verdict": lambda r: FILLS["audit"],
    "Notes": lambda r: FILLS["audit"],
}
n_cols = write_table(ws, groups, match_rows, band_key = "storefront_number", row_fills = row_fills)
verdict_col = get_column_letter(n_cols - 1)
n_matches = len(match_rows)
dv = DataValidation(type = "list", formula1 = '"Correct,Wrong,Unsure"', allow_blank = True)
dv.add(f"{verdict_col}3:{verdict_col}{n_matches + 2}")
ws.add_data_validation(dv)
ws.cell(n_matches + 4, 1, "Source: processed-data/1e_storefront_place_months.parquet (approved rows), 1a_wslcb_locations, 6a_storefront_coverage, veraset_poi_months; built by tools/build_match_audit.py").font = Font(name = FONT, italic = True, size = 9)

# Not approved sheet
ws2 = wb.create_sheet("Considered, not approved")
na_rows = not_approved.to_dicts()
groups2 = [(name, fill, [c for c in cols if c[0] != "Licenses operating in the approved months"])
           for name, fill, cols in groups if name not in ("Visits counted", "Your audit (fill in)")]
groups2 = [("", "id", [("Storefront address (normalized)", 22, lambda r: f"{r['street_address_norm']}, {r['city_norm']}"),
                       ("Why not approved", 26, lambda r: r["why_not"]),
                       ("County", 12, lambda r: r["wslcb_county"])])] + groups2[1:] + [
    ("Months", "visits", [("Months the label is listed at this storefront", 22, lambda r: month_range(r["first_listed"], r["last_listed"], r["months_listed"]))])]
write_table(ws2, groups2, na_rows, band_key = "street_address_norm",
            row_fills = {"Why not approved": lambda r: BAD if r["name_check"] == "no_in_review" else WARN})

# Unmatched sheet
ws3 = wb.create_sheet("Storefronts with no match")
groups3 = [
    ("", "id", [("County", 12, lambda r: r["wslcb_county"]),
                ("Why there is no match", 28, lambda r: r["reason_missing"]),
                ("Months open 2019-2021", 10, lambda r: r["open_months"])]),
    ("WSLCB record (the licensed store)", "wslcb", [
        ("Trade name(s)", 30, lambda r: r["wslcb_tradenames"]),
        ("Street address", 28, lambda r: r["wslcb_address"]),
        ("City", 14, lambda r: title(r["city_norm"])),
        ("ZIP", 8, lambda r: r["wslcb_zip"]),
        ("Licenses", 14, lambda r: r["licenses"]),
        ("Census place of the store", 16, lambda r: r["place_name"] or "(outside any place)")]),
]
write_table(ws3, groups3, unmatched.to_dicts())

# Read me
readme.column_dimensions["A"].width = 34
readme.column_dimensions["B"].width = 110
lines = [
    ("WSLCB-Veraset match audit", None, "h1"),
    ("What this is", "Every link used to count a Veraset visit as a dispensary visit: a WSLCB storefront on one side, the Veraset/SafeGraph place label its visits were attributed to on the other, with the reasons the link was accepted. Check each row and record a verdict.", None),
    ("How to audit a row", "Read across: the blue columns are the licensed store as WSLCB lists it, the green columns are the place in Veraset. Are they the same business at the same location? The yellow 'Why it was matched' columns show which rules it passed. Rows with the same Storefront # are different Veraset labels for one storefront (a renamed listing, a second placekey, a different spelling); each is a separate link, so check each.", None),
    ("Where to write", "Only the bright yellow 'Verdict' and 'Notes' columns at the right of the Matches sheet. Verdict is a drop-down: Correct, Wrong, or Unsure. Example: Verdict = Wrong, Notes = 'Veraset place is the smoke shop next door'.", None),
    ("Approval rule", "A link is approved in a month when all three hold: (1) the address check passes, (2) the name check passes, (3) a WSLCB license at the storefront is listed as operating in the snapshot before or after that month. Visits count only in approved months.", None),
    ("Address check", "Green 'Same address and city': the normalized street address and city are identical. Amber 'Same address and ZIP, city named differently': Veraset used another place name (often a CDP, e.g. Tanglewilde for Olympia). Red 'Different address': found by a looser search (same city and trade name, or same house number and street) and only kept after a manual yes.", None),
    ("Name check", "Green 'Automatic': the Veraset name shares at least half the distinctive words of a WSLCB trade name at that address, and anything else in it is trade-name or cannabis words. Amber 'Manual review: yes': anything else (extra words, cannabis-only names, different addresses), accepted by hand in review/1e_review_page.html.", None),
    ("Name overlap", "The best word overlap with any trade name at the storefront, both ways, ignoring generic words and the city name. 1 = all the distinctive words match.", None),
    ("Building outline", "From the October 2026 SafeGraph Places & Geometry file. Not part of approval; amber when the place shares its outline with a different business, so Veraset may mix their visits.", None),
    ("Approved months / visits", "Months in which the link was approved, and the Veraset visits (num_visits, all devices) counted through it. 'Current WSLCB list' months were approved only because of the undated current license list.", None),
    ("Other sheets", "'Considered, not approved': links that passed the address search but were rejected by hand, or whose name passed but no license was operating in any month. 'Storefronts with no match': operating storefronts (2019-2021) with no approved link, and why.", None),
    ("Counts", None, "h2"),
    ("Links (rows on Matches)", f"=COUNTA(Matches!A3:A{n_matches + 2})", None),
    ("Storefronts with a link", f"=SUMPRODUCT((Matches!B3:B{n_matches + 2}<>\"\")/COUNTIF(Matches!B3:B{n_matches + 2},Matches!B3:B{n_matches + 2}&\"\"))", None),
    ("Links found at a different address", f"=COUNTIF(Matches!R3:R{n_matches + 2},\"Different*\")", None),
    ("Links accepted by manual review", f"=COUNTIF(Matches!S3:S{n_matches + 2},\"Manual*\")", None),
    ("Audit: Correct", f"=COUNTIF(Matches!{verdict_col}3:{verdict_col}{n_matches + 2},\"Correct\")", None),
    ("Audit: Wrong", f"=COUNTIF(Matches!{verdict_col}3:{verdict_col}{n_matches + 2},\"Wrong\")", None),
    ("Audit: Unsure", f"=COUNTIF(Matches!{verdict_col}3:{verdict_col}{n_matches + 2},\"Unsure\")", None),
    ("Audit: not yet checked", f"=B13-B17-B18-B19", None),
    ("Sources", "processed-data/1e_storefront_place_months.parquet ([1e], Step 1 Part E, built by Task 1/step2_wslcb_safegraph_locations_processing.ipynb), 1a_wslcb_locations.parquet ([1a]), 6a_storefront_coverage.parquet ([6a]), and veraset_poi_months.parquet (for the Veraset ZIP). Rebuild with: python3 tools/build_match_audit.py", None),
]
for r, (a, b, kind) in enumerate(lines, start = 1):
    ca, cb = readme.cell(r, 1, a), readme.cell(r, 2, b)
    ca.font = Font(name = FONT, bold = True, size = 16 if kind == "h1" else 12 if kind == "h2" else 10)
    cb.font = Font(name = FONT, size = 10)
    ca.alignment = Alignment(vertical = "top", wrap_text = True)
    cb.alignment = Alignment(vertical = "top", wrap_text = True, horizontal = "left")
    if b and not str(b).startswith("="):
        readme.row_dimensions[r].height = max(15, 14 * (len(b) // 120 + 1))

wb.save(OUT)
print("links:", n_matches, "| storefronts:", matches.select(STOREFRONT).unique().shape[0], "| not approved:", len(na_rows), "| unmatched storefronts:", unmatched.shape[0])
