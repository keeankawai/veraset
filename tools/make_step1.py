import nbformat as nbf
import sys

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""This document aims to accomplish the following:

(a) Merge archived WSLCB store information samples to find plaintext encoding of all locations stores have previously occupied. Use publishing dates by WSLCB to date each sample. Label as ([1a])

(b) Match [1a] against Safegraph information to yield a list of all placekeys belonging to a cannabis retailer. Label as ([1b])

(b') Check that Safegraph's what@where placekey encodings change when the store type at a location changes (ie. a cannabis retailer turning into a gas station should get a new placekey)

(c) For each of the placekeys in [1b], match to a county, city, and zip code, plus CBG if able. Label as ([1c])

(d) Coordinates (and the same census geographies) for every WSLCB location in [1a], matched or not. Label as ([1d])

(e) A month-by-month list of the Veraset place labels that are the cannabis store at each WSLCB storefront, without using distances: exact address, the store's name, and a license operating that month. Whether the building outline is shared with another business is kept as a flag, not used for approval (for now). Label as ([1e])

(e') Every WSLCB storefront and the months a license was operating there, whether or not Veraset has the store (for access measures later). Label as ([1e'])

Irregularities get printed as they come up, and are collected in the markdown cell at the bottom.
""")

code("""### IMPORTS AND PRELIMINARIES
## WORKHORSE LIBRARY
import polars as pl

## EXTRAS
# pandas (with the xlrd dependency) is only needed to read the old .xls WSLCB files, everything else is polars
import pandas as pd
import os
import re

# geopandas is only for Parts C and D (matching coordinates to census boundaries), and urllib grabs the boundary files 
import geopandas as gpd
import urllib.request

# Part D: requests talks to the geocoders, placekey turns a placekey back into the (approximate) coordinates it was built from
import requests
import placekey
import io
import json
import time

# Cleans up some cell outputs
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

pl.Config.set_tbl_rows(30)
pl.Config.set_fmt_str_lengths(40)""")

code("""### HELPERS: TEXT AND ADDRESS NORMALIZATION (copied over from data_loading_processing_week_7)
def normalize_text(expr: pl.Expr) -> pl.Expr:
    return (
        expr
        .str.to_uppercase()
        .str.replace_all(r"[^\\w\\s]", "")
        .str.replace_all(r"\\s+", " ")
        .str.strip_chars()
    )


def normalize_address(expr: pl.Expr) -> pl.Expr:
    expr = normalize_text(expr.str.replace_all("-", " ")) # new: hyphens become spaces first, so "WA-410" turns into "WA 410" instead of "WA410"

    replacements = {
        r"\\bSTREET\\b": "ST",
        r"\\bAVENUE\\b": "AVE",
        r"\\bROAD\\b": "RD",
        r"\\bDRIVE\\b": "DR",
        r"\\bBOULEVARD\\b": "BLVD",
        r"\\bLANE\\b": "LN",
        r"\\bCOURT\\b": "CT",
        r"\\bPLACE\\b": "PL",
        r"\\bPARKWAY\\b": "PKWY",
        r"\\bHIGHWAY\\b": "HWY",
        r"\\bNORTH\\b": "N",
        r"\\bSOUTH\\b": "S",
        r"\\bEAST\\b": "E",
        r"\\bWEST\\b": "W",
        r"\\bNORTHEAST\\b": "NE",
        r"\\bNORTHWEST\\b": "NW",
        r"\\bSOUTHEAST\\b": "SE",
        r"\\bSOUTHWEST\\b": "SW",
        # New for this notebook: state highways get written about six different ways ("SR 20", "STATE ROUTE 20", "State Highway 20", "WA-20", "Washington 20"...)
        r"\\b(STATE ROUTE|STATE RTE|STATE RT|STATE HWY|US HWY)\\b": "HWY",
        r"\\b(SR|WA|US|WASHINGTON|ROUTE|RT)\\s+(\\d+)\\b": "HWY $2",
        r"\\bPKY\\b": "PKWY",
        # ...and so do numbered street names ("FOURTH PLAIN" vs "4TH PLAIN")
        r"\\bFIRST\\b": "1ST",
        r"\\bSECOND\\b": "2ND",
        r"\\bTHIRD\\b": "3RD",
        r"\\bFOURTH\\b": "4TH",
        r"\\bFIFTH\\b": "5TH",
        r"\\bSIXTH\\b": "6TH",
        r"\\bSEVENTH\\b": "7TH",
        r"\\bEIGHTH\\b": "8TH",
        r"\\bNINTH\\b": "9TH",
        r"\\bTENTH\\b": "10TH",
    }

    for old, new in replacements.items():
        expr = expr.str.replace_all(old, new)

    expr = (
        expr
        .str.replace_all(r"\\b(SUITE|STE|UNIT|APT|BLDG|BUILDING|SPC|SPACE)\\s+\\w+\\b", "") # new: a few more unit words
        .str.replace_all(r"\\s+", " ")
        .str.strip_chars()
    )
    # New: unit numbers tacked onto the end with no "STE" in front ("939 N CALLOW AVE 100B", "5200 172ND ST NE F101", "2947 E HWY 101 B").
    # Only stripped right after a street type/direction or a highway number, never a lone N/S/E/W (those are part of the street), and only from
    # addresses with 4+ words so that Pasco/Kennewick's numbered roads ("4525 RD 68") survive
    UNIT = r"(\\s+([ABCDFGHIJKLMOPQRTUVXYZ]|\\w*\\d\\w*)){1,2}$"
    return (
        pl.when(expr.str.count_matches(" ") >= 3)
          .then(expr
                .str.replace(r"\\b(ST|AVE|RD|DR|BLVD|LN|CT|PL|PKWY|WAY|SPEEDWAY|LOOP|CIR|TER|TRL|N|S|E|W|NE|NW|SE|SW)" + UNIT, "$1")
                .str.replace(r"\\b(HWY \\d+)" + UNIT, "$1")
                .str.replace(r"\\bHWY [A-Z]$", "HWY"))
          .otherwise(expr)
    )


# New for this notebook: share of the WSLCB tradename's words that also show up in the Safegraph name (filler words dropped)
FILLER_WORDS = ["LLC", "INC", "THE", "CO", "OF", "AND", "A"]

# Words that half the stores in the state have in their name, so sharing one of them says nothing about being the same store
# (this is how "ANACORTES CANNABIS" ended up matched to "Anacortes Flea Market"). City names get dropped too, row by row
GENERIC_WORDS = ["CANNABIS", "MARIJUANA", "RECREATIONAL", "DISPENSARY", "WEED", "POT", "SHOP", "STORE", "COMPANY", "MARKET", "CLUB", "HOUSE",
                 "RETAIL", "COLLECTIVE", "GREEN", "420", "WA", "WASHINGTON", "N", "S", "E", "W", "NORTH", "SOUTH", "EAST", "WEST"]

def name_overlap(wslcb_name: pl.Expr, safegraph_name: pl.Expr, ignore_words: pl.Expr = None) -> pl.Expr:
    # with ignore_words (a list column, eg. the city's words), generic and city words don't count. Null if nothing distinctive is left in the WSLCB name
    drop = FILLER_WORDS if ignore_words is None else FILLER_WORDS + GENERIC_WORDS
    wslcb_words = wslcb_name.str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(drop)))
    safegraph_words = safegraph_name.str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(drop)))
    if ignore_words is not None:
        wslcb_words = wslcb_words.list.set_difference(ignore_words)
    return (pl.when(wslcb_words.list.len() > 0)
              .then(wslcb_words.list.set_intersection(safegraph_words).list.len() / wslcb_words.list.len())
              .otherwise(None))""")

code("""### PART A, SUBPART I
## Write-up: section 0.2 (the WSLCB samples)
## Output: a long table of every retailer row in every WSLCB sample, one row per license per sample
##         Columns are "snapshot_date" (from the sheet name, NOT the file name, see irregularities), "source_file", "tradename", "license", "ubi",
##         "street_address", "suite", "city", "county", "zip5", "status", "date_field" (DateIssued, or DateCreated in 2018), "license_type"
## Name: 1a_wslcb_snapshots_long.parquet
SNAPSHOT_FILES = sorted(f for f in os.listdir("raw-data") if f.startswith("MarijuanaApplicants"))
COLUMN_NAMES = ["tradename", "license", "ubi", "street_address", "suite", "city", "state", "county", "zip", "license_type", "status", "date_field", "phone"]

results = []
for file in SNAPSHOT_FILES:
    sheets = pd.read_excel(f"raw-data/{file}", sheet_name = None, dtype = str)
    for sheet_name, sheet in sheets.items():
        if "retail" not in sheet_name.lower(): # older files also have producer, processor, and transporter sheets
            continue

        month, day, year = re.search(r"(\\d{1,2})-(\\d{1,2})-(\\d{2})$", sheet_name.strip()).groups()
        print(f"{file:45} -> sheet '{sheet_name}' (header: {list(sheet.columns[[0, 11]])})")

        sheet.columns = COLUMN_NAMES # headers aren't consistent across files (see irregularities), but the column order is
        results.append(
            pl.from_pandas(sheet)
            .with_columns(
                pl.date(2000 + int(year), int(month), int(day)).alias("snapshot_date"),
                pl.lit(file).alias("source_file"),
            )
        )

# The current license list from earlier weeks (no publishing date on it, but it says "CANNABIS" instead of "MARIJUANA", which WA only switched to in 2022)
current = (pl.read_csv("raw-data/WSLCB-licenses.csv", infer_schema = False)
    .select([
        pl.col("Tradename").alias("tradename"),
        pl.col("License").alias("license"),
        pl.col("UBI").alias("ubi"),
        pl.col("Street Address").alias("street_address"),
        pl.col("Suite Rm").alias("suite"),
        pl.col("City").alias("city"),
        pl.col("State").alias("state"),
        pl.col("county"),
        pl.col("Zip Code").alias("zip"),
        pl.col("Priv Desc").alias("license_type"),
        pl.col("Privilege Status").alias("status"),
        pl.lit(None, dtype = pl.String).alias("date_field"),
        pl.col("Day Phone").alias("phone"),
    ])
    .with_columns(
        pl.lit(None, dtype = pl.Date).alias("snapshot_date"), # null = the undated, post-2022 list
        pl.lit("WSLCB-licenses.csv").alias("source_file"),
    )
)

snapshots_long = (pl.concat(results + [current], how = "vertical")
    .with_columns(pl.col(pl.String).str.strip_chars()) # everything comes padded with spaces
    .with_columns(
        pl.col("license").cast(pl.Int64), # some licenses have leading zeros in the .xls files but not in the .csv
        pl.col("zip").str.slice(0, 5).alias("zip5"),
        pl.col("county").str.to_uppercase(),
    )
    .select(["snapshot_date", "source_file", "tradename", "license", "ubi", "street_address", "suite", "city", "county", "zip5", "status", "date_field", "license_type"])
    .sort(["snapshot_date", "license"], nulls_last = True)
)

snapshots_long.write_parquet("processed-data/1a_wslcb_snapshots_long.parquet")

snapshots_long.group_by(["snapshot_date", "source_file"]).agg(pl.len(), pl.col("license").n_unique().alias("num_licenses")).sort("snapshot_date", nulls_last = True)""")

code("""### PART A, SUBPART II
## Write-up: section 0.2 (which statuses count as operating, OPERATING_STATUSES)
## Output: a table of every location a cannabis retailer license has been listed at, one row per license per (normalized) address
##         Columns are "license", "street_address_norm", "city_norm", plus everything we know about that license at that address: tradenames, UBIs,
##         statuses, county, zip, the first and last sample it showed up in ("first_seen", "last_seen"), and the first and last sample where the status
##         says the store was actually up and running ("first_operating", "last_operating")
## Name: 1a_wslcb_locations.parquet
# Statuses that mean a license was issued and attached to a physical store at the time of the sample
OPERATING_STATUSES = ["ACTIVE (ISSUED)", "PENDING (ISSUED)", "CLOSED (TEMPORARY)"]

locations = (snapshots_long
    .with_columns(
        street_address_norm = normalize_address(pl.col("street_address")),
        city_norm = normalize_text(pl.col("city")),
        is_operating = pl.col("status").is_in(OPERATING_STATUSES),
    )
    .group_by(["license", "street_address_norm", "city_norm"])
    .agg(
        pl.col("tradename").unique().sort().alias("tradenames"),
        pl.col("ubi").drop_nulls().unique().sort().alias("ubis"),
        pl.col("status").unique().sort().alias("statuses"),
        pl.col("street_address").first(), # one raw version of the address, for reference
        pl.col("county").drop_nulls().mode().first().alias("county"),
        pl.col("zip5").drop_nulls().mode().first().alias("zip5"),
        pl.col("snapshot_date").min().alias("first_seen"),
        pl.col("snapshot_date").max().alias("last_seen"),
        pl.col("snapshot_date").filter(pl.col("is_operating")).min().alias("first_operating"),
        pl.col("snapshot_date").filter(pl.col("is_operating")).max().alias("last_operating"),
        pl.col("snapshot_date").drop_nulls().n_unique().alias("num_dated_snapshots"),
        pl.col("snapshot_date").is_null().any().alias("in_current_list"),
        pl.col("status").filter(pl.col("snapshot_date").is_null()).unique().sort().alias("current_list_statuses"),
        pl.col("is_operating").any().alias("ever_operating"),
    )
    .with_columns(
        pl.col("street_address_norm").str.contains("TITLE CERTIFICATE").alias("is_placeholder_address"), # title certificate holders have no real address
    )
    .sort(["license", "first_seen"])
)

locations.write_parquet("processed-data/1a_wslcb_locations.parquet")

print(locations.shape, "| licenses:", locations["license"].n_unique(), "| placeholder addresses:", locations["is_placeholder_address"].sum(),
      "| never operating:", (~locations["ever_operating"]).sum())
locations""")

code("""### IRREGULARITIES, PART A
# (1) File names vs. the dates on the sheets inside them
print(snapshots_long.filter(pl.col("snapshot_date").is_not_null()).select(["source_file", "snapshot_date"]).unique().sort("snapshot_date"))

# (2) Licenses listed more than once in the same sample
dup_rows = snapshots_long.filter(pl.len().over(["snapshot_date", "license"]) > 1)
print("\\nlicenses listed more than once within a sample:", dup_rows.group_by("source_file").agg(pl.col("license").n_unique()).rows())

# (3) Licenses showing up at more than one address, under more than one name, or under more than one UBI (ownership changes)
per_license = locations.filter(~pl.col("is_placeholder_address")).group_by("license").agg(pl.len().alias("num_addresses"))
print("licenses at 2+ addresses:", per_license.filter(pl.col("num_addresses") > 1).shape[0])
print("licenses with 2+ tradenames:", snapshots_long.group_by("license").agg(pl.col("tradename").n_unique()).filter(pl.col("tradename") > 1).shape[0])
print("licenses with 2+ UBIs:", snapshots_long.group_by("license").agg(pl.col("ubi").drop_nulls().n_unique()).filter(pl.col("ubi") > 1).shape[0])

# (4) Addresses with more than one license number on them (ie. new license after a sale, or an applicant re-filing)
per_address = locations.filter(~pl.col("is_placeholder_address")).group_by(["street_address_norm", "city_norm"]).agg(pl.col("license").n_unique().alias("num_licenses"))
print("addresses with 2+ licenses:", per_address.filter(pl.col("num_licenses") > 1).shape[0])

# (5) Licenses in the dated samples but not the current list, and vice versa
dated_ids = snapshots_long.filter(pl.col("snapshot_date").is_not_null())["license"].unique()
current_ids = snapshots_long.filter(pl.col("snapshot_date").is_null())["license"].unique()
print("licenses only in the dated samples:", (~dated_ids.is_in(current_ids.implode())).sum(), "| only in the current list:", (~current_ids.is_in(dated_ids.implode())).sum())

# (6) Does "DateIssued" behave like an issue date? (if it does, it shouldn't change between samples for the same license)
date_changes = (snapshots_long
    .filter(pl.col("snapshot_date").is_not_null() & ~pl.col("date_field").is_in(["00000000", ""]))
    .group_by("license").agg(pl.col("date_field").n_unique())
)
print("licenses whose DateIssued changes between samples:", date_changes.filter(pl.col("date_field") > 1).shape[0], "of", date_changes.shape[0])

# (7) Licenses that were operating in one sample and are just gone from the next one (no "CLOSED" row, nothing). Later files seem to drop
#     long-closed licenses, so disappearing from the list doesn't tell us much on its own
dated = sorted(d for d in snapshots_long["snapshot_date"].unique().to_list() if d is not None)
for earlier, later in zip(dated, dated[1:]):
    earlier_df = snapshots_long.filter(pl.col("snapshot_date") == earlier)
    later_df = snapshots_long.filter(pl.col("snapshot_date") == later)
    vanished_operating = earlier_df.filter(pl.col("status").is_in(OPERATING_STATUSES)).join(later_df, on = "license", how = "anti")
    print(f"{earlier} -> {later}: operating licenses that vanish: {vanished_operating['license'].n_unique():3} | any licenses that vanish: {earlier_df.join(later_df, on = 'license', how = 'anti')['license'].n_unique()}")

# (8) Status counts by sample
snapshots_long.sort("snapshot_date", nulls_last = True).group_by(["snapshot_date", "status"], maintain_order = True).len().pivot(on = "snapshot_date", index = "status", values = "len").fill_null(0)""")

code("""# A few examples of licenses moving around / changing names, for a sense of what's going on
multi_location_ids = per_license.filter(pl.col("num_addresses") > 1)["license"].head(4).implode()
(snapshots_long
    .filter(pl.col("license").is_in(multi_location_ids))
    .select(["license", "snapshot_date", "tradename", "street_address", "city", "status"])
    .sort(["license", "snapshot_date"], nulls_last = True)
)""")

code("""### INTERMEDIATE WORK: MONTHLY POI INFORMATION FROM VERASET
## Output: one row per placekey per month per version of its attributes, with the number of visits it got
##         Columns are "placekey", "safegraph_place_id", "location_name", "naics_code", "street_address", "city", "zip_code", "census_block_group",
##         "num_visits", "month" (safegraph_place_id is Safegraph's older place ID, which Veraset fills in even when the placekey is blank, see Part E)
## Name: veraset_poi_months.parquet
# Veraset attaches Safegraph's POI info to every visit as of when the visit happened, so this doubles as a month-by-month history of each placekey.
# Done month by month because a single pass over everything runs out of memory
MONTHS = sorted(folder for folder in os.listdir("raw-data/veraset-visits-wa") if folder.startswith("month="))

results = []
for MONTH in MONTHS:
    month_df = (pl.scan_parquet(f"raw-data/veraset-visits-wa/{MONTH}/*.parquet")
        .filter(pl.col("location_name") != "home") # home visits have no placekey
        .group_by(["placekey", "safegraph_place_id", "location_name", "naics_code", "street_address", "city", "zip_code", "census_block_group"])
        .agg(pl.len().alias("num_visits"))
        .with_columns(pl.lit(MONTH.replace("month=", "")).alias("month"))
        .collect(engine = "streaming")
    )
    results.append(month_df)

veraset_pois = pl.concat(results)
veraset_pois.write_parquet("processed-data/veraset_poi_months.parquet")

print(veraset_pois.shape, "| distinct placekeys:", veraset_pois["placekey"].n_unique())""")

code("""### PART B
## Output: a table of every placekey belonging to a cannabis retailer, one row per [1a] location per placekey
##         Columns are the [1a] keys ("license", "street_address_norm", "city_norm"), "placekey", "poi_source" (safegraph or veraset), the POI's
##         name/category/open/close info, "match_pass" (which of the 4 passes found it), "name_overlap", and "match_type":
##           "name"          - the POI name shares at least half the words of the WSLCB tradename
##           "cannabis_name" - no name match, but the POI name is obviously a cannabis store (has "cannabis", "marijuana", "kush" etc. in it)
##           "category"      - neither of those, but it's filed under 424210, which is where Safegraph puts most cannabis stores
##           "related"       - none of those, but filed under a nearby retail category (misc. store retailers, tobacco stores). Only kept if nothing better
##                             turned up for that location, since most of these are the smoke shop next door
##         "match_pass" 1-4 are the week 7 passes. 5-7 are fallbacks for locations that 1-4 didn't find a cannabis store for, and they only accept
##         cannabis-looking places (cannabis-y name or category 424210):
##           5 - same house number, city and street name, spelled differently (catches what the address normalizer still misses)
##           6 - in the same placekey "where" (building/lot) as something that IS at the WSLCB address, and on the same street or with a matching name
##               (catches neighboring street numbers)
##           7 - same as 5 but in any city (catches postal city vs. actual city, eg. Lake Forest Park vs. Shoreline)
##         Anything else matched at the same address (restaurants next door, etc.) is dropped here, but kept in 1b_match_candidates_all.parquet
## Name: 1b_cannabis_placekeys.parquet
# (1) Load in Safegraph data and align column names with Veraset (as in week 7, plus the extra columns needed for checking)
safegraph_lf = (pl.scan_parquet(
        "raw-data/SafeGraph-POI-Metadata/*.parquet",
    )
    .select([pl.col("CITY").alias("city"),
             pl.col("LOCATION_NAME").alias("location_name"),
             pl.col("NAICS_CODE").alias("naics_code"),
             pl.col("SUB_CATEGORY").alias("sub_category"),
             pl.col("PLACEKEY").alias("placekey"),
             pl.col("STREET_ADDRESS").alias("street_address"),
             pl.col("LATITUDE").alias("latitude"),
             pl.col("LONGITUDE").alias("longitude"),
             pl.col("OPENED_ON").alias("opened_on"),
             pl.col("CLOSED_ON").alias("closed_on")])
)

CANNABIS_NAICS = ["424210"] # "Drugs and Druggists' Sundries Merchant Wholesalers" is where Safegraph files most dispensaries
RELATED_NAICS = ["453998", "453991", "459999", "459991"] # misc. store retailers and tobacco stores (2017 and 2022 NAICS codes)
CANNABIS_WORDS = r"CANNABIS|MARIJUANA|DISPENSARY|\\bCANNA|\\bPOT\\b|\\bWEED\\b|\\bKUSH(\\b|21)|\\bBUDS?\\b|\\b420\\b" # "KUSH" alone would also catch "Kushner"

# (1.5) The Safegraph file is missing about a quarter of the placekeys that show up in Veraset (including a lot of the cannabis ones), so every
#       Veraset placekey that isn't in Safegraph gets added from Veraset's own POI info. Each version of its name/address is kept so renamed stores can still match
veraset_only_lf = (veraset_pois.lazy()
    .join(safegraph_lf.select("placekey"), on = "placekey", how = "anti")
    .group_by(["placekey", "location_name", "street_address", "city"])
    .agg(pl.col("naics_code").sort_by("num_visits").last()) # most-visited NAICS code for that version
)

poi_lf = pl.concat([
    safegraph_lf.with_columns(pl.lit("safegraph").alias("poi_source")),
    veraset_only_lf.with_columns(pl.lit("veraset").alias("poi_source")),
], how = "diagonal_relaxed") # columns Veraset doesn't have (lat/lon, open/close dates) are left null

# (2) Normalizing both sides. Each distinct license / name / address combination from [1a] gets matched (placeholder addresses can't be)
# Note: "safegraph_norm" keeps its week 7 name, but now holds the Veraset-only POIs too
safegraph_norm = (
    poi_lf
    .with_columns(
        city_norm=normalize_text(pl.col("city")),
        location_name_norm=normalize_text(pl.col("location_name")),
        street_address_norm=normalize_address(pl.col("street_address")),
    )
)

wslcb_norm = (snapshots_long.lazy()
    .with_columns(
        city_norm=normalize_text(pl.col("city")),
        location_name_norm=normalize_text(pl.col("tradename")),
        street_address_norm=normalize_address(pl.col("street_address")),
    )
    .filter(~pl.col("street_address_norm").str.contains("TITLE CERTIFICATE"))
    .select(["license", "location_name_norm", "street_address_norm", "city_norm"])
    .unique()
)""")

code("""# (3) The four matching passes from week 7, unchanged except that each one records which pass it came from
match1 = (
    wslcb_norm
    .join(
        safegraph_norm.select(
            "street_address_norm",
            "city_norm",
            "placekey",
        ),
        on=["street_address_norm", "city_norm"],
        how="inner",
    )
    .with_columns(pl.lit(1).alias("match_pass"))
)

remaining1 = (
    wslcb_norm
    .join(
        match1.select(
            "street_address_norm",
            "city_norm",
        ).unique(),
        on=["street_address_norm", "city_norm"],
        how="anti",
    )
)

match2 = (
    remaining1
    .join(
        safegraph_norm.select(
            "street_address_norm",
            "location_name_norm",
            "placekey",
        ),
        on=[
            "street_address_norm",
            "location_name_norm",
        ],
        how="inner",
    )
    .with_columns(pl.lit(2).alias("match_pass"))
)

remaining2 = (
    remaining1
    .join(
        match2.select(
            "street_address_norm",
            "location_name_norm",
        ).unique(),
        on=[
            "street_address_norm",
            "location_name_norm",
        ],
        how="anti",
    )
)

unique_name_city = (
    safegraph_norm
    .group_by("location_name_norm", "city_norm")
    .len()
    .filter(pl.col("len") == 1)
    .drop("len")
)

safegraph_unique_name_city = (
    safegraph_norm
    .join(
        unique_name_city,
        on=["location_name_norm", "city_norm"],
        how="inner",
    )
)

match3 = (
    remaining2
    .join(
        safegraph_unique_name_city.select(
            "location_name_norm",
            "city_norm",
            "placekey",
        ),
        on=["location_name_norm", "city_norm"],
        how="inner",
    )
    .with_columns(pl.lit(3).alias("match_pass"))
)

remaining3 = (
    remaining2
    .join(
        match3.select(
            "street_address_norm",
            "city_norm",
            "location_name_norm",
        ).unique(),
        on=[
            "street_address_norm",
            "city_norm",
            "location_name_norm",
        ],
        how="anti",
    )
)

match4 = (
    remaining3
    .join(
        safegraph_norm.select(
            "location_name_norm",
            "city_norm",
            "placekey",
        ),
        on=["location_name_norm", "city_norm"],
        how="inner",
    )
    .with_columns(pl.lit(4).alias("match_pass"))
)

matches = pl.concat(
    [
        match1,
        match2,
        match3,
        match4,
    ],
    how="vertical",
)""")

code("""# (4) Attaching the POI info to every candidate and sorting out which ones are actually the cannabis store
#     (Veraset-only placekeys can have several versions, so they can show up more than once here. The collapsing step below keeps the best one)
POI_INFO = ["placekey", "poi_source", "location_name", "naics_code", "sub_category", "street_address", "opened_on", "closed_on", "latitude", "longitude"]

def classify(matched):
    return (matched
        .join(safegraph_norm.select(POI_INFO + [pl.col("location_name_norm").alias("safegraph_name_norm")]), on = "placekey", how = "left")
        .with_columns(
            name_overlap(pl.col("location_name_norm"), pl.col("safegraph_name_norm"), pl.col("city_norm").str.split(" ")).alias("name_overlap"),
        )
        .with_columns(
            pl.when(pl.col("name_overlap") >= 0.5).then(pl.lit("name"))
              .when(pl.col("safegraph_name_norm").str.contains(CANNABIS_WORDS)).then(pl.lit("cannabis_name"))
              .when(pl.col("naics_code").is_in(CANNABIS_NAICS)).then(pl.lit("category"))
              .when(pl.col("naics_code").is_in(RELATED_NAICS)).then(pl.lit("related"))
              .otherwise(pl.lit("other"))
              .alias("match_type")
        )
    )

candidates = classify(matches).collect()
print("passes 1-4 candidate rows:", candidates.shape[0], "| by match type:", candidates["match_type"].value_counts().sort("match_type").rows())""")

code("""# (5) Fallback passes for every location that passes 1-4 didn't find a real cannabis match for (nothing, or only "related"/"other" places)
LOCATION_KEY = ["license", "street_address_norm", "city_norm"]
GOOD_TYPES = ["name", "cannabis_name", "category"]

# Street name words without the house number, street types, or directions, and "415TH" written as "415" (so it matches "415 AVE SE")
STREET_TYPES = ["ST", "AVE", "RD", "DR", "BLVD", "LN", "CT", "PL", "PKWY", "WAY", "HWY", "LOOP", "CIR", "N", "S", "E", "W", "NE", "NW", "SE", "SW"]

def street_words(address: pl.Expr) -> pl.Expr:
    return (address.str.split(" ").list.slice(1)
        .list.eval(pl.element().str.replace(r"^(\\d+)(ST|ND|RD|TH)$", "$1").filter(~pl.element().is_in(STREET_TYPES)))
        .alias("street_words"))

def same_street(left: str, right: str) -> pl.Expr:
    return pl.col(left).list.set_intersection(pl.col(right)).list.len() > 0

poi_lookup = (safegraph_norm
    .with_columns(
        pl.col("placekey").str.split("@").list.last().alias("where"),
        pl.col("street_address_norm").str.extract(r"^(\\d+)").alias("house_number"),
        street_words(pl.col("street_address_norm")),
    )
    .filter(pl.col("location_name_norm").str.contains(CANNABIS_WORDS) | pl.col("naics_code").is_in(CANNABIS_NAICS)) # cannabis-looking places only
    .select(["placekey", "where", "house_number", "city_norm", "street_words"])
    .unique()
    .collect()
)

needs_help = (wslcb_norm.collect()
    .join(candidates.filter(pl.col("match_type").is_in(GOOD_TYPES)).select(LOCATION_KEY).unique(), on = LOCATION_KEY, how = "anti")
    .with_columns(
        pl.col("street_address_norm").str.extract(r"^(\\d+)").alias("house_number"),
        street_words(pl.col("street_address_norm")),
    )
)
print("locations needing a fallback:", needs_help.select(LOCATION_KEY).unique().shape[0])

MATCH_COLUMNS = ["license", "location_name_norm", "street_address_norm", "city_norm", "placekey", "match_pass"]

# Pass 5: same house number, same city, and at least one street word in common (house number + city alone pulled in "Top Pot Doughnuts")
match5 = (needs_help
    .join(poi_lookup.select(["placekey", "house_number", "city_norm", pl.col("street_words").alias("poi_street_words")]), on = ["house_number", "city_norm"], how = "inner")
    .filter(same_street("street_words", "poi_street_words"))
    .with_columns(pl.lit(5).alias("match_pass"))
    .select(MATCH_COLUMNS)
)

# Pass 6: same "where" as anything passes 1-4 found at the address (whatever kind of business it was). A "where" can cover a whole block, so this
#         also needs a street word in common or the names to match (checked further down, once the names are attached)
address_wheres = (candidates
    .join(needs_help.select(LOCATION_KEY).unique(), on = LOCATION_KEY, how = "semi")
    .filter(pl.col("placekey").is_not_null())
    .select(LOCATION_KEY + [pl.col("placekey").str.split("@").list.last().alias("where")])
    .unique()
)
match6 = (needs_help
    .join(address_wheres, on = LOCATION_KEY, how = "inner")
    .join(poi_lookup.select(["placekey", "where", pl.col("street_words").alias("poi_street_words")]), on = "where", how = "inner")
    .with_columns(pl.lit(6).alias("match_pass"), same_street("street_words", "poi_street_words").alias("same_street"))
    .select(MATCH_COLUMNS + ["same_street"])
)

fallback56 = (classify(pl.concat([match5.with_columns(pl.lit(True).alias("same_street")), match6]).lazy())
    .filter(pl.col("same_street") | (pl.col("match_type") == "name"))
    .drop("same_street")
    .collect()
)

# Pass 7: same as pass 5 but in any city. Only for locations 5 and 6 didn't help
still_missing = needs_help.join(fallback56.filter(pl.col("match_type").is_in(GOOD_TYPES)).select(LOCATION_KEY).unique(), on = LOCATION_KEY, how = "anti")
match7 = (still_missing
    .join(poi_lookup.select(["placekey", "house_number", pl.col("street_words").alias("poi_street_words")]), on = "house_number", how = "inner")
    .filter(same_street("street_words", "poi_street_words"))
    .with_columns(pl.lit(7).alias("match_pass"))
    .select(MATCH_COLUMNS)
)

fallback_candidates = pl.concat([fallback56, classify(match7.lazy()).collect()], how = "vertical").unique()
print("fallback rows by pass:", fallback_candidates["match_pass"].value_counts().sort("match_pass").rows())

candidates = pl.concat([candidates, fallback_candidates], how = "vertical")
candidates.write_parquet("processed-data/1b_match_candidates_all.parquet")""")

code("""# (6) Collapsing the different tradenames of a license at an address (keeping the best match for each placekey), and dropping "related" matches
#     wherever the same location has a better one (that's the smoke shop next to the dispensary case)
MATCH_TYPE_ORDER = {"name": 0, "cannabis_name": 1, "category": 2, "related": 3, "other": 4}

cannabis_placekeys = (candidates
    .filter(pl.col("match_type") != "other")
    .sort([pl.col("match_type").replace_strict(MATCH_TYPE_ORDER), "match_pass", pl.col("name_overlap")], descending = [False, False, True], nulls_last = True)
    .group_by(["license", "street_address_norm", "city_norm", "placekey"], maintain_order = True)
    .agg(
        pl.col("match_type").first(),
        pl.col("match_pass").first(),
        pl.col("name_overlap").max(),
        pl.col("location_name_norm").unique().sort().alias("wslcb_names_norm"),
        pl.col(POI_INFO[1:]).first(),
    )
    .filter((pl.col("match_type") != "related") | ~pl.col("match_type").is_in(GOOD_TYPES).any().over(LOCATION_KEY))
    .sort(["license", "street_address_norm", pl.col("match_type").replace_strict(MATCH_TYPE_ORDER)])
)

cannabis_placekeys.write_parquet("processed-data/1b_cannabis_placekeys.parquet")

print("[1b] by match type:", cannabis_placekeys["match_type"].value_counts().sort("match_type").rows(), "| by pass:", cannabis_placekeys["match_pass"].value_counts().sort("match_pass").rows())
print("[1b] by POI source:", cannabis_placekeys.unique("placekey")["poi_source"].value_counts().sort("poi_source").rows())
print("[1b] rows:", cannabis_placekeys.shape[0], "| distinct placekeys:", cannabis_placekeys["placekey"].n_unique(),
      "| licenses covered:", cannabis_placekeys["license"].n_unique(), "of", snapshots_long["license"].n_unique())
cannabis_placekeys""")

code("""### IRREGULARITIES, PART B
# (1) How many [1a] locations got at least one cannabis placekey, split by whether they were ever operating
coverage = (locations
    .filter(~pl.col("is_placeholder_address"))
    .join(cannabis_placekeys.group_by(["license", "street_address_norm", "city_norm"]).agg(pl.len().alias("num_placekeys")),
          on = ["license", "street_address_norm", "city_norm"], how = "left")
    .with_columns(pl.col("num_placekeys").fill_null(0))
)
print(coverage.group_by("ever_operating").agg(pl.len().alias("num_locations"), (pl.col("num_placekeys") > 0).sum().alias("num_matched")).sort("ever_operating"))

# (2) How many placekeys each matched location ends up with (several is normal, one per NAICS code)
print(coverage.filter(pl.col("num_placekeys") > 0)["num_placekeys"].value_counts().sort("num_placekeys").rows())

# (3) Placekeys claimed by more than one license (same storefront, different licenses over time, or two licenses at one address)
shared = cannabis_placekeys.group_by("placekey").agg(pl.col("license").n_unique().alias("num_licenses")).filter(pl.col("num_licenses") > 1)
print("placekeys claimed by 2+ licenses:", shared.shape[0])

# (4) Comparison against week 7 (637 licenses matched with the current list only, 400 with any Veraset visits). Week 7 kept every candidate,
#     including the "other" ones (restaurants next door etc.), so the like-for-like comparison is the all-candidates number
veraset_placekeys = veraset_pois.select("placekey").unique()
print("licenses with any candidate (week 7 style):", candidates["license"].n_unique(),
      "| licenses with a cannabis placekey:", cannabis_placekeys["license"].n_unique(),
      "| ...of which have Veraset visits:", cannabis_placekeys.join(veraset_placekeys, on = "placekey", how = "semi")["license"].n_unique())

# (5) Veraset placekeys missing from the Safegraph file entirely (can't be matched through Safegraph at all)
safegraph_placekeys = safegraph_lf.select("placekey").collect()
print("Veraset placekeys not in the Safegraph file:", veraset_placekeys.join(safegraph_placekeys, on = "placekey", how = "anti").shape[0], "of", veraset_placekeys.shape[0])

# (6) Operating locations that never got matched
coverage.filter(pl.col("ever_operating") & (pl.col("num_placekeys") == 0)).select(["license", "tradenames", "street_address", "city_norm", "first_operating", "last_operating"])""")

code("""# (7) The reverse: places that look like cannabis stores (cannabis-y name AND a cannabis/related category) that didn't get matched to any WSLCB location.
#     These are either stores missing from all the WSLCB samples (eg. opened after Apr 2021 and since closed) or ones whose address got mangled somewhere
unmatched_lookalikes = (safegraph_norm
    .filter(pl.col("naics_code").is_in(CANNABIS_NAICS + RELATED_NAICS) & pl.col("location_name_norm").str.contains(CANNABIS_WORDS))
    .join(cannabis_placekeys.select("placekey").lazy(), on = "placekey", how = "anti")
    .select(["placekey", "poi_source", "location_name", "street_address", "city", "opened_on", "closed_on"])
    .unique("placekey")
    .sort("city")
    .collect()
)
print("unmatched cannabis-looking places:", unmatched_lookalikes.shape[0], "| by source:", unmatched_lookalikes["poi_source"].value_counts().sort("poi_source").rows())
unmatched_lookalikes""")

code("""### PART B'
## Checking whether placekeys change when the business at a location changes. A placekey is "what@where": the "where" half is the physical spot,
## the "what" half is supposed to be the business at that spot. If that works as advertised, a placekey should never switch from a cannabis store
## into something else. The Safegraph file only has one row per placekey (no history), so the history comes from Veraset's monthly POI info
# (1) For each placekey and month, the name and NAICS code it went by (the most-visited version if there's more than one that month)
monthly_versions = (veraset_pois
    .group_by(["placekey", "month"])
    .agg(
        pl.col("location_name").sort_by("num_visits").last(),
        pl.col("naics_code").sort_by("num_visits").last(),
        pl.col("num_visits").sum(),
    )
    .sort(["placekey", "month"])
    .with_columns(
        pl.col("location_name").shift().over("placekey").alias("prev_location_name"),
        pl.col("naics_code").shift().over("placekey").alias("prev_naics_code"),
        pl.col("month").shift().over("placekey").alias("prev_month"),
    )
)

changes = monthly_versions.filter(
    pl.col("prev_month").is_not_null() & ((pl.col("naics_code") != pl.col("prev_naics_code")) | (pl.col("location_name") != pl.col("prev_location_name")))
)

# (2) The general picture across all of Veraset
print("placekeys in Veraset:", monthly_versions["placekey"].n_unique())
print("placekeys that change NAICS code at some point:", changes.filter(pl.col("naics_code") != pl.col("prev_naics_code"))["placekey"].n_unique())
print("placekeys that change name at some point:", changes.filter(pl.col("location_name") != pl.col("prev_location_name"))["placekey"].n_unique())
print("most common months for changes (Safegraph data releases?):", changes.group_by("month").len().sort("len", descending = True).head(4).rows())

# Name changes that aren't just formatting (the normalized names share no words at all)
real_name_changes = (changes
    .filter(pl.col("location_name") != pl.col("prev_location_name"))
    .with_columns(name_overlap(normalize_text(pl.col("prev_location_name")), normalize_text(pl.col("location_name"))).alias("overlap"))
    .filter(pl.col("overlap") == 0)
)
print("placekeys whose name changes to something completely different:", real_name_changes["placekey"].n_unique())
real_name_changes.select(["placekey", "prev_month", "month", "prev_location_name", "location_name", "prev_naics_code", "naics_code"]).sample(15, seed = 42)""")

code("""# (3) Now just the cannabis placekeys from [1b]: does any of them ever go by a non-cannabis name / category in Veraset?
cannabis_ids = cannabis_placekeys.select("placekey").unique()

cannabis_history = (monthly_versions
    .join(cannabis_ids, on = "placekey", how = "semi")
    .with_columns(
        (pl.col("naics_code").is_in(CANNABIS_NAICS + RELATED_NAICS) | normalize_text(pl.col("location_name")).str.contains(CANNABIS_WORDS)).alias("looks_cannabis")
    )
)

cannabis_changes = (changes
    .join(cannabis_ids, on = "placekey", how = "semi")
    .with_columns(name_overlap(normalize_text(pl.col("prev_location_name")), normalize_text(pl.col("location_name"))).alias("overlap"))
)

print("[1b] placekeys seen in Veraset:", cannabis_history["placekey"].n_unique(), "of", cannabis_ids.shape[0])
print("...that change NAICS code:", cannabis_changes.filter(pl.col("naics_code") != pl.col("prev_naics_code"))["placekey"].n_unique())
print("...that change to a completely different name:", cannabis_changes.filter(pl.col("overlap") == 0)["placekey"].n_unique())
print("...that go by a non-cannabis name AND category in some month:", cannabis_history.filter(~pl.col("looks_cannabis"))["placekey"].n_unique())

# The NAICS changes are mostly noise: the same store, same name, bouncing between unrelated categories from one Safegraph release to the next
print("\\nmost common NAICS switches among [1b] placekeys:")
print(cannabis_changes.filter(pl.col("naics_code") != pl.col("prev_naics_code")).group_by(["prev_naics_code", "naics_code"]).len().sort("len", descending = True).head(8))

# The name changes are the real test: these are the cases where the business under a cannabis placekey turned into something with a different name
cannabis_changes.filter(pl.col("overlap") == 0).select(["placekey", "prev_month", "month", "prev_location_name", "location_name", "prev_naics_code", "naics_code"])""")

code("""# (4) Same "where", different "what": everything Safegraph has at the same physical spots as the [1b] placekeys. If placekeys work as advertised,
#     a store that closed and got replaced should show up here as a separate placekey with a different "what" half (and its own open/close dates)
where_parts = cannabis_placekeys.select(pl.col("placekey").str.split("@").list.last().alias("where")).unique()

same_where = (safegraph_lf
    .with_columns(pl.col("placekey").str.split("@").list.last().alias("where"))
    .join(where_parts.lazy(), on = "where", how = "semi")
    .join(cannabis_ids.lazy().with_columns(pl.lit(True).alias("in_1b")), on = "placekey", how = "left")
    .with_columns(pl.col("in_1b").fill_null(False))
    .select(["where", "placekey", "in_1b", "location_name", "naics_code", "opened_on", "closed_on"])
    .sort(["where", "opened_on"], nulls_last = True)
    .collect()
)

per_where = same_where.group_by("where").agg(pl.len().alias("num_placekeys"), pl.col("in_1b").sum().alias("num_in_1b"))
print("distinct 'where's among [1b] placekeys:", per_where.shape[0], "| with other businesses at the same spot:", per_where.filter(pl.col("num_placekeys") > pl.col("num_in_1b")).shape[0])
print("placekeys per 'where':", per_where["num_placekeys"].value_counts().sort("num_placekeys").rows())

# Can open/close dates show a replacement business getting its own placekey? Only if Safegraph actually filled the dates in
print("same-'where' placekeys with an opened_on date:", same_where["opened_on"].is_not_null().sum(), "of", same_where.shape[0],
      "| with a closed_on date:", same_where["closed_on"].is_not_null().sum())
print("[1b] placekeys with a closed_on date:", same_where.filter(pl.col("in_1b"))["closed_on"].is_not_null().sum(), "of", same_where["in_1b"].sum())

# Spots where a [1b] placekey closed and something else opened afterwards (ie. the replacement got its own placekey, as it should)
replacements = (same_where.filter(pl.col("in_1b") & pl.col("closed_on").is_not_null()).select(["where", pl.col("closed_on").alias("cannabis_closed_on")])
    .join(same_where.filter(~pl.col("in_1b") & pl.col("opened_on").is_not_null()), on = "where")
    .filter(pl.col("opened_on") >= pl.col("cannabis_closed_on"))
)
print("closed cannabis spots with a later business under a new placekey:", replacements["where"].n_unique())
replacements.head(15)""")

code("""### PART C 
## Output: a table of the [1b] placekeys with the census geographies they sit in 
##         Columns are "placekey", "poi_source", "latitude", "longitude", "geo_method", "cbg2020", "cbg2010" (Veraset's CBG vintage), "county_fips", 
##         "county_name", "place_fips", "place_name", "place_type" (city / town / CDP, null if it's in no place), "zcta", plus Veraset's own "veraset_zip_code" 
##         and "veraset_cbg2010" (most-visited version) and the WSLCB "wslcb_zip5" / "wslcb_county", for comparison 
##         geo_method is "point" (Safegraph coordinates matched into the boundary files) or "veraset_cbg" (no coordinates, so Veraset's 2010 CBG is used, 
##         with the place/ZCTA taken from a point inside that CBG, which is rougher) 
## Name: 1c_cannabis_placekey_geography.parquet 
# (1) Grabbing the census boundary files (only downloaded once, then cached in raw-data/census) 
TIGER_FILES = [
    "TIGER2020/BG/tl_2020_53_bg.zip", 
    "TIGER2010/BG/2010/tl_2010_53_bg10.zip", 
    "TIGER2020/PLACE/tl_2020_53_place.zip", 
    "TIGER2020/ZCTA520/tl_2020_us_zcta520.zip", # national file (~500 MB), only the part around WA gets read in below 
]
os.makedirs("raw-data/census", exist_ok = True) 
for path in TIGER_FILES: 
    local_path = "raw-data/census/" + path.split("/")[-1] 
    if not os.path.exists(local_path): 
        urllib.request.urlretrieve("https://www2.census.gov/geo/tiger/" + path, local_path) 

bg2020_shapes = gpd.read_file("zip://raw-data/census/tl_2020_53_bg.zip")[["GEOID", "geometry"]].rename(columns = {"GEOID": "cbg2020"}) 
bg2010_shapes = gpd.read_file("zip://raw-data/census/tl_2010_53_bg10.zip")[["GEOID10", "geometry"]].rename(columns = {"GEOID10": "cbg2010"}).to_crs(bg2020_shapes.crs) 
place_shapes = gpd.read_file("zip://raw-data/census/tl_2020_53_place.zip")[["GEOID", "geometry"]].rename(columns = {"GEOID": "place_fips"}) 
zcta_shapes = (gpd.read_file("zip://raw-data/census/tl_2020_us_zcta520.zip", bbox = tuple(bg2020_shapes.total_bounds)) # bbox = WA's bounding box 
    [["ZCTA5CE20", "geometry"]].rename(columns = {"ZCTA5CE20": "zcta"})) 

print("CRS:", bg2020_shapes.crs.name, "| shapes read in:", len(bg2020_shapes), len(bg2010_shapes), len(place_shapes), len(zcta_shapes))""")

code("""# (2) One row per [1b] placekey, plus what Veraset and WSLCB say about where it is 
placekey_info = cannabis_placekeys.group_by("placekey").agg(pl.col(["poi_source", "latitude", "longitude"]).first()) 

veraset_location = (veraset_pois 
    .join(placekey_info.select("placekey"), on = "placekey", how = "semi") 
    .group_by(["placekey", "zip_code", "census_block_group"]) 
    .agg(pl.col("num_visits").sum()) 
    .sort("num_visits", descending = True) 
    .group_by("placekey", maintain_order = True) 
    .agg(
        pl.col("zip_code").first().alias("veraset_zip_code"), 
        pl.col("census_block_group").first().alias("veraset_cbg2010"), 
    )
)

wslcb_location = (cannabis_placekeys 
    .join(locations.select(["license", "street_address_norm", "city_norm", "county", "zip5"]), on = ["license", "street_address_norm", "city_norm"], how = "left") 
    .group_by("placekey") 
    .agg(
        pl.col("zip5").drop_nulls().mode().first().alias("wslcb_zip5"), 
        pl.col("county").drop_nulls().mode().first().alias("wslcb_county"), 
    )
)

placekey_info = (placekey_info 
    .join(veraset_location, on = "placekey", how = "left") 
    .join(wslcb_location, on = "placekey", how = "left") 
)

# (3) Turning each placekey into a point: Safegraph's coordinates if it has them, otherwise a point inside its Veraset CBG 
with_coords = placekey_info.filter(pl.col("latitude").is_not_null()).to_pandas() 
points = gpd.GeoDataFrame(with_coords, geometry = gpd.points_from_xy(with_coords["longitude"], with_coords["latitude"]), crs = "EPSG:4326").to_crs(bg2020_shapes.crs) 
points["geo_method"] = "point" 

without_coords = placekey_info.filter(pl.col("latitude").is_null()).to_pandas().merge(bg2010_shapes, left_on = "veraset_cbg2010", right_on = "cbg2010", how = "left") 
cbg_points = gpd.GeoDataFrame(without_coords.drop(columns = ["cbg2010"]), geometry = "geometry", crs = bg2020_shapes.crs) 
cbg_points["geometry"] = cbg_points.geometry.representative_point() # a point guaranteed to be inside the CBG (unlike the centroid) 
cbg_points["geo_method"] = "veraset_cbg" 

all_points = pd.concat([points, cbg_points[~cbg_points.geometry.is_empty & cbg_points.geometry.notna()]], ignore_index = True) 
print("placekeys:", len(placekey_info), "| with Safegraph coordinates:", len(points), "| placed using their Veraset CBG:", (all_points["geo_method"] == "veraset_cbg").sum(), 
      "| no location at all:", len(placekey_info) - len(all_points)) 

# (4) Point-in-polygon for each layer (dropping duplicates in case a point sits exactly on a border) 
for shapes in [bg2020_shapes, bg2010_shapes, place_shapes, zcta_shapes]: 
    all_points = gpd.sjoin(all_points, shapes, how = "left", predicate = "within").drop(columns = ["index_right"]).drop_duplicates("placekey") 

# (5) Back to polars, with county/place names from the Step 3 tables so everything lines up 
counties = pl.read_parquet("processed-data/3c_county_population_2020.parquet").select(["county_fips", "county_name"]) 
places = pl.read_parquet("processed-data/3d_place_population_2020.parquet").select(["place_fips", "place_name", "place_type"]) 

placekey_geography = (pl.from_pandas(pd.DataFrame(all_points.drop(columns = ["geometry"]))) 
    .with_columns(
        pl.when(pl.col("geo_method") == "veraset_cbg").then(pl.col("veraset_cbg2010")).otherwise(pl.col("cbg2010")).alias("cbg2010"), # keep Veraset's own CBG when that's what we started from 
    )
    .with_columns(pl.coalesce(pl.col("cbg2020"), pl.col("cbg2010")).str.slice(0, 5).alias("county_fips")) 
    .join(counties, on = "county_fips", how = "left") 
    .join(places, on = "place_fips", how = "left") 
    .select(["placekey", "poi_source", "latitude", "longitude", "geo_method", "cbg2020", "cbg2010", "county_fips", "county_name", "place_fips", "place_name", "place_type", 
             "zcta", "veraset_zip_code", "veraset_cbg2010", "wslcb_zip5", "wslcb_county"]) 
    .sort("placekey")
)

placekey_geography.write_parquet("processed-data/1c_cannabis_placekey_geography.parquet") 

placekey_geography""")

code("""### IRREGULARITIES, PART C 
# (1) How often the different sources agree (only counting placekeys where both sides have a value) 
checks = placekey_geography.filter(pl.col("geo_method") == "point") 
def agreement(left, right): 
    both = checks.filter(pl.col(left).is_not_null() & pl.col(right).is_not_null()) 
    return f"{(both[left] == both[right]).sum()} of {both.shape[0]}" 

print("Safegraph point's 2010 CBG == Veraset's CBG:  ", agreement("cbg2010", "veraset_cbg2010")) 
print("Safegraph point's ZCTA == Veraset's zip code: ", agreement("zcta", "veraset_zip_code")) 
print("Safegraph point's ZCTA == WSLCB zip:          ", agreement("zcta", "wslcb_zip5")) 
print("county name == WSLCB county:                   ", 
      (placekey_geography["county_name"].str.to_uppercase() == placekey_geography["wslcb_county"]).sum(), "of", placekey_geography["wslcb_county"].is_not_null().sum()) 

# (2) Points that fall outside WA's boundaries entirely, and dispensaries outside any census place (unincorporated areas) 
print("no 2020 CBG (outside WA?):", placekey_geography["cbg2020"].is_null().sum(), "| not in any place:", placekey_geography["place_fips"].is_null().sum(), 
      "| place types:", placekey_geography["place_type"].value_counts().sort("place_type").rows()) 

# (3) The disagreements on county, for a look 
placekey_geography.filter(pl.col("county_name").str.to_uppercase() != pl.col("wslcb_county")).select(["placekey", "geo_method", "county_name", "wslcb_county", "place_name", "latitude", "longitude"])""")

code("""### PART D
## Write-up: sections 4-5 (where each store is: county, city, ZCTA, CBG)
## Output: coordinates for every WSLCB location in [1a] (except the title certificate placeholders), whether or not it matched a placekey
##         Columns are the [1a] keys ("license", "street_address_norm", "city_norm"), "street_address", "zip5", "ever_operating", "latitude", "longitude",
##         "geo_method", "geo_matched_address", plus the same census geographies as [1c] ("cbg2020", "cbg2010", "county_fips", "county_name", "place_fips",
##         "place_name", "place_type", "zcta")
##         geo_method, best first:
##           "same_address_poi"       - median Safegraph coordinates of whatever businesses share the address (any kind of business, it's just for location).
##                                      Safegraph's points sit on the building, so they beat the Census geocoder when there are any
##           "census_exact" / "census_nonexact" - the Census Bureau's geocoder (2020 TIGER address ranges, so the point is interpolated along the block face,
##                                      which can be a few km off on long rural highways, and "nonexact" sometimes lands on the wrong street, eg. W vs E)
##           "same_address_placekey"  - center of the "where" hexagon of a Veraset placekey at the same address (the hexagons are ~65m across)
##           "matched_placekey"       - coordinates of the [1b] placekey matched to it (only from the fallback passes, so a neighboring address)
##           "openstreetmap"          - OpenStreetMap's Nominatim geocoder
##           "zcta"                   - a point inside the WSLCB zip code's ZCTA (last resort, rough)
## Name: 1d_wslcb_location_coordinates.parquet
# (1) The Census Bureau's batch geocoder (free, no key, up to 10,000 addresses per call). The response is cached in raw-data/census so this only runs once
to_geocode = (locations
    .filter(~pl.col("is_placeholder_address"))
    .with_row_index("geocode_id")
    # the cached response gets matched back on the address text it echoes, not on geocode_id, because [1a]'s row order can change between runs
    # when a license has two addresses with the same first_seen
    .with_columns((pl.col("street_address") + ", " + pl.col("city_norm") + ", WA, " + pl.col("zip5").fill_null("")).alias("census_input"))
)

CENSUS_GEOCODE_FILE = "raw-data/census/wslcb_census_geocoder.csv"
if not os.path.exists(CENSUS_GEOCODE_FILE):
    response = requests.post(
        "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch",
        data = {"benchmark": "Public_AR_Census2020", "vintage": "Census2020_Census2020"},
        files = {"addressFile": ("addresses.csv", to_geocode.select(["geocode_id", "street_address", "city_norm", pl.lit("WA"), "zip5"]).write_csv(include_header = False))},
        timeout = 900,
    )
    open(CENSUS_GEOCODE_FILE, "w").write(response.text)

census_geocoded = (pl.read_csv(CENSUS_GEOCODE_FILE, has_header = False, truncate_ragged_lines = True, infer_schema = False,
                               new_columns = ["geocode_id", "input_address", "match", "match_type", "geo_matched_address", "lon_lat", "tiger_line", "side", "state", "county", "tract", "block"])
    .filter(pl.col("match") == "Match")
    .select(
        pl.col("input_address").alias("census_input"),
        pl.col("lon_lat").str.split(",").list.get(1).cast(pl.Float64).alias("latitude"),
        pl.col("lon_lat").str.split(",").list.get(0).cast(pl.Float64).alias("longitude"),
        ("census_" + pl.col("match_type").str.to_lowercase().str.replace("_", "")).alias("geo_method"),
        "geo_matched_address",
    )
    .unique("census_input")
)
print("Census geocoder:", census_geocoded["geo_method"].value_counts().sort("geo_method").rows(), "of", to_geocode.shape[0])

# (2) Fallbacks for what the Census geocoder misses (mostly highway addresses and newer roads that TIGER doesn't have address ranges for).
#     Anything at the same address works for a location, even if it's the restaurant next door
LOCATION_KEY = ["license", "street_address_norm", "city_norm"]

same_address_poi = (safegraph_norm
    .filter(pl.col("latitude").is_not_null())
    .group_by(["street_address_norm", "city_norm"])
    .agg(pl.col("latitude").median(), pl.col("longitude").median(), pl.col("street_address").first().alias("geo_matched_address"))
    .with_columns(pl.lit("same_address_poi").alias("geo_method"))
    .collect()
)

same_address_placekey = (veraset_pois
    .select(["placekey", "street_address", "city"]).drop_nulls("placekey").unique()
    .with_columns(normalize_address(pl.col("street_address")).alias("street_address_norm"), normalize_text(pl.col("city")).alias("city_norm"))
    .sort(["street_address_norm", "city_norm", "placekey", "street_address"])
    .unique(["street_address_norm", "city_norm"], keep = "first", maintain_order = True)
    .join(to_geocode.select(["street_address_norm", "city_norm"]).unique(), on = ["street_address_norm", "city_norm"], how = "semi") # only the addresses we need, decoding is slow-ish
    .with_columns(pl.col("placekey").map_elements(lambda key: list(placekey.placekey_to_geo(key)), return_dtype = pl.List(pl.Float64)).alias("lat_lon"))
    .select(["street_address_norm", "city_norm", pl.col("lat_lon").list.get(0).alias("latitude"), pl.col("lat_lon").list.get(1).alias("longitude"),
             pl.col("street_address").alias("geo_matched_address"), pl.lit("same_address_placekey").alias("geo_method")])
)

matched_placekey = (cannabis_placekeys
    .filter(pl.col("latitude").is_not_null())
    .sort("match_pass")
    .group_by(LOCATION_KEY, maintain_order = True)
    .agg(pl.col("latitude").first(), pl.col("longitude").first(), pl.col("street_address").first().alias("geo_matched_address"))
    .with_columns(pl.lit("matched_placekey").alias("geo_method"))
)

GEO_COLUMNS = ["latitude", "longitude", "geo_method", "geo_matched_address"]
coordinates = (to_geocode
    .join(same_address_poi, on = ["street_address_norm", "city_norm"], how = "left")
    .join(census_geocoded.rename({c: c + "_2" for c in GEO_COLUMNS}), on = "census_input", how = "left")
    .join(same_address_placekey.rename({c: c + "_3" for c in GEO_COLUMNS}), on = ["street_address_norm", "city_norm"], how = "left")
    .join(matched_placekey.rename({c: c + "_4" for c in GEO_COLUMNS}), on = LOCATION_KEY, how = "left")
    .with_columns(
        # whichever source comes first wins, and all four columns come from that same source
        pl.when(pl.col("latitude").is_not_null()).then(0)
          .when(pl.col("latitude_2").is_not_null()).then(2)
          .when(pl.col("latitude_3").is_not_null()).then(3)
          .when(pl.col("latitude_4").is_not_null()).then(4)
          .alias("source")
    )
    .with_columns([
        pl.when(pl.col("source") == 0).then(pl.col(c))
          .when(pl.col("source") == 2).then(pl.col(c + "_2"))
          .when(pl.col("source") == 3).then(pl.col(c + "_3"))
          .when(pl.col("source") == 4).then(pl.col(c + "_4"))
          .alias(c)
        for c in GEO_COLUMNS
    ])
)

# (3) OpenStreetMap for whatever's left (Nominatim asks for at most one request a second, also cached so it only runs once)
NOMINATIM_FILE = "raw-data/census/wslcb_nominatim.json"
nominatim = json.load(open(NOMINATIM_FILE)) if os.path.exists(NOMINATIM_FILE) else {}
for row in coordinates.filter(pl.col("latitude").is_null()).iter_rows(named = True):
    query = f"{row['street_address']}, {row['city_norm'].title()}, WA {row['zip5'] or ''}"
    if query not in nominatim:
        response = requests.get("https://nominatim.openstreetmap.org/search", params = {"q": query, "format": "json", "limit": 1, "countrycodes": "us"},
                                headers = {"User-Agent": "regulation-through-convenience research project (WSLCB store geocoding)"}, timeout = 60)
        nominatim[query] = response.json()
        time.sleep(1.1)
json.dump(nominatim, open(NOMINATIM_FILE, "w"))

osm_points = pl.DataFrame(
    [{"query": q, "latitude_5": float(r[0]["lat"]), "longitude_5": float(r[0]["lon"]), "geo_matched_address_5": r[0]["display_name"]} for q, r in nominatim.items() if r],
    schema = {"query": pl.String, "latitude_5": pl.Float64, "longitude_5": pl.Float64, "geo_matched_address_5": pl.String},
)
coordinates = (coordinates
    .with_columns((pl.col("street_address") + ", " + pl.col("city_norm").str.to_titlecase() + ", WA " + pl.col("zip5").fill_null("")).alias("query"))
    .join(osm_points, on = "query", how = "left")
    .with_columns(
        pl.when(pl.col("latitude").is_null() & pl.col("latitude_5").is_not_null()).then(pl.lit("openstreetmap")).otherwise(pl.col("geo_method")).alias("geo_method"),
        pl.coalesce("latitude", "latitude_5").alias("latitude"),
        pl.coalesce("longitude", "longitude_5").alias("longitude"),
        pl.coalesce("geo_matched_address", "geo_matched_address_5").alias("geo_matched_address"),
    )
)
print("still missing after OpenStreetMap:", coordinates["latitude"].is_null().sum())""")

code("""# (4) Census geographies, the same way as Part C. Anything still without coordinates gets a point inside its WSLCB zip's ZCTA
with_coords = coordinates.filter(pl.col("latitude").is_not_null()).select(LOCATION_KEY + GEO_COLUMNS).to_pandas()
points = gpd.GeoDataFrame(with_coords, geometry = gpd.points_from_xy(with_coords["longitude"], with_coords["latitude"]), crs = "EPSG:4326").to_crs(bg2020_shapes.crs)

without_coords = coordinates.filter(pl.col("latitude").is_null()).select(LOCATION_KEY + ["zip5"]).to_pandas().merge(zcta_shapes, left_on = "zip5", right_on = "zcta", how = "inner")
zcta_points = gpd.GeoDataFrame(without_coords.drop(columns = ["zip5", "zcta"]), geometry = "geometry", crs = bg2020_shapes.crs)
zcta_points["geometry"] = zcta_points.geometry.representative_point()
zcta_points["geo_method"] = "zcta"
zcta_points["geo_matched_address"] = None
lat_lon = zcta_points.to_crs("EPSG:4326").geometry
zcta_points["latitude"], zcta_points["longitude"] = lat_lon.y, lat_lon.x

all_points = pd.concat([points, zcta_points], ignore_index = True)
for shapes in [bg2020_shapes, bg2010_shapes, place_shapes, zcta_shapes]:
    all_points = gpd.sjoin(all_points, shapes, how = "left", predicate = "within").drop(columns = ["index_right"]).drop_duplicates(LOCATION_KEY)

location_coordinates = (to_geocode.select(LOCATION_KEY + ["street_address", "zip5", "ever_operating"])
    .join(pl.from_pandas(pd.DataFrame(all_points.drop(columns = ["geometry"]))), on = LOCATION_KEY, how = "left")
    .with_columns(pl.coalesce(pl.col("cbg2020"), pl.col("cbg2010")).str.slice(0, 5).alias("county_fips"))
    .join(counties, on = "county_fips", how = "left")
    .join(places, on = "place_fips", how = "left")
    .select(LOCATION_KEY + ["street_address", "zip5", "ever_operating", "latitude", "longitude", "geo_method", "geo_matched_address",
                            "cbg2020", "cbg2010", "county_fips", "county_name", "place_fips", "place_name", "place_type", "zcta"])
    .sort(LOCATION_KEY)
)

location_coordinates.write_parquet("processed-data/1d_wslcb_location_coordinates.parquet")

print("[1d] rows:", location_coordinates.shape[0], "| by method:", location_coordinates["geo_method"].value_counts().sort("count", descending = True).rows(),
      "| no coordinates:", location_coordinates["latitude"].is_null().sum())
location_coordinates""")

code("""### IRREGULARITIES, PART D
# (1) How far the Census geocoder's point is from Safegraph's point for the same store (only [1b] matches at the exact WSLCB address), ie. how much
#     worse things would be if the Census geocoder were all we had.
#     The Census geocoder interpolates along the street, so some distance is normal, especially on long rural blocks
def distance_m(lat1, lon1, lat2, lon2):
    # haversine, in meters
    lat1, lon1, lat2, lon2 = [c.radians() for c in [lat1, lon1, lat2, lon2]]
    a = ((lat2 - lat1) / 2).sin() ** 2 + lat1.cos() * lat2.cos() * ((lon2 - lon1) / 2).sin() ** 2
    return 2 * 6371000 * a.sqrt().arcsin()

geocode_check = (to_geocode.select(["census_input"] + LOCATION_KEY + ["street_address"])
    .join(census_geocoded, on = "census_input", how = "inner")
    .join(cannabis_placekeys.filter((pl.col("match_pass") == 1) & pl.col("latitude").is_not_null())
            .group_by(LOCATION_KEY).agg(pl.col("latitude").median().alias("sg_latitude"), pl.col("longitude").median().alias("sg_longitude"), pl.col("location_name").first()),
          on = LOCATION_KEY, how = "inner")
    .with_columns(distance_m(pl.col("latitude"), pl.col("longitude"), pl.col("sg_latitude"), pl.col("sg_longitude")).alias("distance_m"))
)
print("Census point vs Safegraph point, same store:", geocode_check.shape[0], "locations | median distance:", round(geocode_check["distance_m"].median()), "m",
      "| 90th percentile:", round(geocode_check["distance_m"].quantile(0.9)), "m | over 250m:", (geocode_check["distance_m"] > 250).sum())

# (2) The worst ones, for a look
geocode_check.sort("distance_m", descending = True).select(["license", "street_address", "city_norm", "geo_method", "geo_matched_address", "location_name", "distance_m"]).head(10)""")

code("""### PART E
## Write-up: section 0.2 (which visits count as dispensary visits) and section 5.1 ([1e'], open stores)
## Output: every version of every Veraset place label at (or, pending review, possibly at) a WSLCB storefront, month by month, and whether visits to it
##         count as visits to the cannabis store. Only "approved" rows should be used for counting dispensary visits ("outline_evidence" for robustness checks)
##         Columns are the label key ("month", "safegraph_place_id", "location_name", "street_address", "city"; join the raw visits on these to filter
##         them, see the irregularities), "placekey" (filled in from other months when Veraset left it blank, see "placekey_filled_in"), "naics_code",
##         "num_visits", the storefront ("street_address_norm", "city_norm", "tradenames", "operating_licenses"), "address_match" ("exact",
##         "same_address_and_zip" when Veraset calls the city something else, or "different_address" for review candidates), the two checks, and a flag:
##           name_rule / name_check - is the label the store? "tradename" (shares the distinctive words of a tradename used at that address) is accepted
##                                    automatically ("auto") at the exact address (same city, or same zip) when there's nothing else in the name, or the
##                                    rest is cannabis words.
##                                    Everything else that could be the store ("tradename" with extra words, "cannabis_name", "category_only", anything
##                                    at a different address) is "review" until it gets a manual verdict ("yes_in_review" / "no_in_review")
##           wslcb_operating        - was a license at that address listed as operating in the WSLCB sample before or after that month?
##           outline_evidence       - from the Oct 2026 Places & Geometry file, does the place have a building outline to itself?
##                                      "own"           - its own outline, or one shared only with other listings of the same store
##                                      "shared"        - an outline shared with a different business (eg. a dispensary and the physical therapist
##                                                        upstairs). Veraset may not tell visits to those apart
##                                      "likely_shared" - not in the 2026 file (closed), and something at its address shares an outline now
##                                      "likely_own"    - not in the 2026 file, and everything at its address now has its own outline
##                                      "unknown"       - no outline (just a point), or not in the 2026 file and nothing at its address either
##         approved = name_check is "auto" or "yes_in_review", and wslcb_operating. outline_evidence isn't part of it (for now); filter on it as well
##         ("own", "likely_own", "unknown") for the stricter version
##         No distances anywhere: a visit only counts through the place Veraset itself attributed it to
## Name: 1e_storefront_place_months.parquet
## Output: [1e'], every storefront-month with a license operating there by the WSLCB samples, Veraset or not
##         Columns are "street_address_norm", "city_norm", "month" ("YYYY-MM"), "operating_licenses", "current_list_only" (see below)
## Name: 1e_storefront_months.parquet
# (1) Storefronts (normalized address + city, across all the licenses ever operating there) and the months each license was operating at one
STOREFRONT = ["street_address_norm", "city_norm"]
SAMPLE_DATES = sorted(d for d in snapshots_long["snapshot_date"].unique().to_list() if d is not None)

# The WSLCB sample right before (or during) each month, and the one right after it. After Apr 2021 the "after" sample is the undated current list
month_samples = []
for month in [folder.replace("month=", "") for folder in MONTHS]:
    month_end = pl.Series([month]).str.to_date("%Y-%m").dt.month_end().item()
    before = max(d for d in SAMPLE_DATES if d <= month_end)
    later = [d for d in SAMPLE_DATES if d > before]
    month_samples.append({"month": month, "sample_before": str(before), "sample_after": str(later[0]) if later else "current"})
month_samples = pl.DataFrame(month_samples)

license_samples = (snapshots_long
    .with_columns(
        street_address_norm = normalize_address(pl.col("street_address")),
        city_norm = normalize_text(pl.col("city")),
        sample = pl.col("snapshot_date").cast(pl.String).fill_null("current"),
    )
    .filter(~pl.col("street_address_norm").str.contains("TITLE CERTIFICATE"))
    .group_by(["license"] + STOREFRONT + ["sample"])
    .agg(pl.col("status").is_in(OPERATING_STATUSES).any().alias("is_operating"))
    .filter(pl.col("is_operating"))
)

# A license counts as operating at an address in a month if it's listed as operating there in the sample before OR the sample after.
# The current list has no date (2022 or later), so a license that only shows up as operating there might have opened years after 2021. Those
# months are marked "current_list_only": good enough to accept a visit with a matching name, but left out of the match rate denominators
license_months = (month_samples
    .unpivot(index = "month", on = ["sample_before", "sample_after"], value_name = "sample")
    .join(license_samples, on = "sample", how = "inner")
    .group_by(["license"] + STOREFRONT + ["month"])
    .agg((pl.col("sample") == "current").all().alias("current_list_only"))
)

storefronts = (locations
    .filter(pl.col("ever_operating") & ~pl.col("is_placeholder_address"))
    .group_by(STOREFRONT)
    .agg(pl.col("tradenames").list.explode().unique().sort())
)
storefront_months = (license_months
    .group_by(STOREFRONT + ["month"])
    .agg(pl.col("license").unique().sort().alias("operating_licenses"), pl.col("current_list_only").all())
)

storefront_months.join(storefronts, on = STOREFRONT, how = "semi").sort(STOREFRONT + ["month"]).write_parquet("processed-data/1e_storefront_months.parquet") # [1e']

active_license_months = license_months.filter(~pl.col("current_list_only"))
print("licenses operating in some month of 2019-2021:", active_license_months["license"].n_unique(), "of", locations["license"].n_unique(),
      "(plus", license_months["license"].n_unique() - active_license_months["license"].n_unique(), "only by the current list)",
      "| storefronts:", active_license_months.select(STOREFRONT).unique().shape[0], "of", storefronts.shape[0],
      "| storefront-months:", active_license_months.select(STOREFRONT + ["month"]).unique().shape[0])""")

code("""# (2) Match rates, used after every step below. Denominators are the licenses and storefronts ("stores") operating in some month of 2019-2021 (by
#     the dated samples), and those storefronts' operating months. A license / storefront counts as matched if, in at least one month it was
#     operating, its storefront has a row that passes. Rows in parentheses are what-ifs, not steps
match_rate_log = []

def match_rates(passing, label):
    matched = active_license_months.join(passing.select(STOREFRONT + ["month"]).unique(), on = STOREFRONT + ["month"], how = "semi")
    num_licenses = active_license_months["license"].n_unique()
    num_storefronts = active_license_months.select(STOREFRONT).unique().shape[0]
    num_storefront_months = active_license_months.select(STOREFRONT + ["month"]).unique().shape[0]
    matched_storefronts = matched.select(STOREFRONT).unique().shape[0]
    matched_storefront_months = matched.select(STOREFRONT + ["month"]).unique().shape[0]
    match_rate_log.append({
        "step": label,
        "licenses": matched["license"].n_unique(),
        "pct_lic": round(100 * matched["license"].n_unique() / num_licenses, 1),
        "stores": matched_storefronts,
        "pct_stores": round(100 * matched_storefronts / num_storefronts, 1),
        "store_months": matched_storefront_months,
        "pct_months": round(100 * matched_storefront_months / num_storefront_months, 1),
        "visits": passing["num_visits"].sum(),
    })
    return pl.DataFrame(match_rate_log)""")

code("""# (3) Every version of every place's label in each month. Veraset's legacy "safegraph_place_id" is never blank, while "placekey" is blank on 1.7M
#     visits (mostly Nov 2020 - Jul 2021, see irregularities), so the place ID is the key and the placekey gets filled in from the other months
LABEL_KEY = ["month", "safegraph_place_id", "location_name", "street_address", "city"]

placekey_lookup = (veraset_pois
    .filter(pl.col("placekey").is_not_null())
    .group_by(["safegraph_place_id", "placekey"]).agg(pl.col("num_visits").sum())
    .sort("num_visits")
    .group_by("safegraph_place_id").agg(pl.col("placekey").last().alias("lookup_placekey")) # most-visited placekey, for the rare IDs with 2+
)

place_labels = (veraset_pois
    .group_by(LABEL_KEY)
    .agg(
        pl.col("placekey").sort_by("num_visits").drop_nulls().last(),
        pl.col("naics_code").sort_by("num_visits").last(),
        pl.col("zip_code").sort_by("num_visits").last(),
        pl.col("num_visits").sum(),
    )
    .join(placekey_lookup, on = "safegraph_place_id", how = "left")
    .with_columns(
        pl.col("placekey").is_null().alias("placekey_filled_in"),
        pl.coalesce("placekey", "lookup_placekey").alias("placekey"),
        normalize_address(pl.col("street_address")).alias("street_address_norm"),
        normalize_text(pl.col("city")).alias("city_norm"),
        normalize_text(pl.col("location_name")).alias("location_name_norm"),
    )
    .drop("lookup_placekey")
)

# Exact address, and either the same city or the same zip code. Veraset often names the unincorporated area instead of the postal city ("tanglewilde"
# for Olympia, "parkland" for Tacoma, "summit view" for Puyallup), so the zip stands in for the city there. A place one door down, upstairs with a
# different address, or across the parking lot never makes it in
storefront_zips = (locations
    .filter(pl.col("ever_operating") & ~pl.col("is_placeholder_address"))
    .select(STOREFRONT + ["zip5"])
    .drop_nulls("zip5")
    .unique()
)
by_city = place_labels.with_columns(pl.lit("exact").alias("address_match"))
by_zip = (place_labels
    .join(storefront_zips, left_on = ["street_address_norm", "zip_code"], right_on = ["street_address_norm", "zip5"], how = "inner", suffix = "_storefront")
    .filter(pl.col("city_norm") != pl.col("city_norm_storefront"))
    .with_columns(pl.col("city_norm_storefront").alias("city_norm"), pl.lit("same_address_and_zip").alias("address_match"))
    .drop("city_norm_storefront")
)
at_storefronts = (pl.concat([by_city, by_zip])
    .join(storefronts, on = STOREFRONT, how = "inner")
    .join(storefront_months, on = STOREFRONT + ["month"], how = "left")
    .with_columns(pl.col("operating_licenses").is_not_null().alias("wslcb_operating"), pl.col("current_list_only").fill_null(False))
)
print("place labels at a storefront address:", at_storefronts.shape[0], "| places:", at_storefronts["safegraph_place_id"].n_unique(),
      "| by city / by zip:", at_storefronts["address_match"].value_counts().sort("address_match").rows(),
      "| placekeys filled in:", (at_storefronts["placekey_filled_in"] & at_storefronts["placekey"].is_not_null()).sum(),
      "| still no placekey:", at_storefronts["placekey"].is_null().sum())
match_rates(at_storefronts.filter(pl.col("wslcb_operating")), "exact address, any business")""")

code("""# (4) Is the label the store? Name overlap goes both ways (share of the tradename's distinctive words in the Veraset name, and the other way around),
#     so "2020 Solutions" still matches "2020 CANNABIS SOLUTIONS MT BAKER HIGHWAY" but "Baker Bear Grocery" doesn't
#     Write-up: section 0.2 (the name check, and what gets accepted without review)
CANNABIS_NAME_WORDS = r"CANNABIS|MARIJUANA|DISPENSARY|\\bCANNA\\b|\\bWEED\\b|\\bKUSH|\\b420\\b|\\bTHC\\b|\\bPOT SHOP\\b|\\bGANJA" # stricter than Part B's (no "POT", "BUDS")
STORE_WORDS = CANNABIS_NAME_WORDS + r"|RECREATIONAL|\\bPOT\\b|\\bBUDS?\\b|\\bMJ\\b|\\b21\\b|\\bI502\\b|\\b502\\b" # only used on top of a tradename match
HEAD_SHOP_WORDS = r"\\bGLASS\\b|SMOKE|VAPE|TOBACCO|HOOKAH|CLOTHING|APPAREL|\\bCBD\\b|\\bHEMP\\b" # the smoke shop next door, often with "KUSH" or "420" in the name

def tradename_overlap(rows, group_key):
    # best overlap with any of the storefront's tradenames, both ways, plus the share of the label's own words that come from a tradename
    return (rows
        .explode("tradenames")
        .with_columns(normalize_text(pl.col("tradenames")).alias("tradename_norm"), pl.col("city_norm").str.split(" ").alias("city_words"))
        .with_columns(
            name_overlap(pl.col("tradename_norm"), pl.col("location_name_norm"), pl.col("city_words")).alias("tradename_in_label"),
            name_overlap(pl.col("location_name_norm"), pl.col("tradename_norm"), pl.col("city_words")).alias("label_in_tradename"),
        )
        .group_by(group_key)
        .agg(
            pl.max_horizontal("tradename_in_label", "label_in_tradename").max().alias("name_overlap"),
            pl.col("tradename_in_label").max(),
            pl.col("label_in_tradename").max(),
        )
    )

def check_names(rows):
    return (rows
        .join(tradename_overlap(rows.select(LABEL_KEY + STOREFRONT + ["location_name_norm", "tradenames"]), LABEL_KEY + STOREFRONT),
              on = LABEL_KEY + STOREFRONT, how = "left", nulls_equal = True)
        .with_columns(
            pl.when(pl.col("name_overlap") >= 0.5).then(pl.lit("tradename"))
              .when(pl.col("location_name_norm").str.contains(CANNABIS_NAME_WORDS)).then(pl.lit("cannabis_name"))
              .when(pl.col("naics_code") == "424210").then(pl.lit("category_only"))
              .otherwise(pl.lit("other_business"))
              .alias("name_rule"),
        )
        .with_columns(
            # accepted without review: a tradename match at the exact address where the rest of the label is nothing but more tradename or
            # cannabis words (so no "Origins Coffee Roasters" for "ORIGINS", or "Gypsy Glass" for "GYPSY GREEN"). Smoke shop words only count
            # against a label when the tradename doesn't have them too ("CANNABIS AND GLASS" is a real dispensary)
            pl.when(pl.col("address_match").is_in(["exact", "same_address_and_zip"]) & (pl.col("name_rule") == "tradename")
                    & (~pl.col("location_name_norm").str.contains(HEAD_SHOP_WORDS) | normalize_text(pl.col("tradenames").list.join(" ")).str.contains(HEAD_SHOP_WORDS))
                    & ((pl.col("label_in_tradename") == 1) | pl.col("location_name_norm").str.contains(STORE_WORDS))).then(pl.lit("auto"))
              .when(pl.col("name_rule") == "other_business").then(pl.lit("no"))
              .otherwise(pl.lit("review"))
              .alias("name_check")
        )
    )

at_storefronts = check_names(at_storefronts)
print(at_storefronts.group_by(["name_rule", "name_check"]).agg(pl.col("safegraph_place_id").n_unique().alias("places"), pl.len().alias("labels"), pl.col("num_visits").sum()).sort(["name_rule", "name_check"]))

NAME_OK = ["auto", "yes_in_review"]
match_rates(at_storefronts.filter(pl.col("wslcb_operating") & pl.col("name_check").is_in(NAME_OK)), "+ store name (auto)")""")

code("""# (5) Candidates at a different address, for storefronts where nothing at the exact address could be the store (Veraset writes the address differently,
#     or uses the postal city). Never approved automatically, only after a manual look. Two ways in:
#       - same city, the label has at least half the tradename's distinctive words, and it's either nothing but tradename words or has a cannabis
#         word in it. Places sitting at another WSLCB storefront's exact address are left out (that's just the other branch of a chain)
#       - same house number and street name (any city) and a cannabis-looking name or tradename match
#     Write-up: section 0.2 (the manual review and the verdicts file)
LABEL_ID = ["safegraph_place_id", "location_name", "street_address", "city"]
no_store_at_address = (storefronts
    .join(storefront_months.select(STOREFRONT).unique(), on = STOREFRONT, how = "semi")
    .join(at_storefronts.filter(pl.col("name_check") != "no").select(STOREFRONT).unique(), on = STOREFRONT, how = "anti")
    .with_columns(pl.col("street_address_norm").str.extract(r"^(\\d+)").alias("house_number"), street_words(pl.col("street_address_norm")))
)
distinct_labels = (place_labels
    .select(LABEL_ID + ["location_name_norm", pl.col("street_address_norm").alias("label_address_norm"), pl.col("city_norm").alias("label_city_norm")])
    .unique(LABEL_ID)
    .with_columns(pl.col("label_address_norm").str.extract(r"^(\\d+)").alias("house_number"),
                  street_words(pl.col("label_address_norm")).alias("label_street_words"))
)

same_city = (no_store_at_address.select(STOREFRONT + ["tradenames"])
    .join(distinct_labels.join(storefronts.select(pl.col("street_address_norm").alias("label_address_norm"), pl.col("city_norm").alias("label_city_norm")),
                               on = ["label_address_norm", "label_city_norm"], how = "anti"),
          left_on = "city_norm", right_on = "label_city_norm", how = "inner")
)
same_city = (same_city
    .join(tradename_overlap(same_city.select(LABEL_ID + STOREFRONT + ["location_name_norm", "tradenames"]), LABEL_ID + STOREFRONT), on = LABEL_ID + STOREFRONT, how = "inner")
    .filter((pl.col("tradename_in_label") >= 0.5) & ~pl.col("location_name_norm").str.contains(HEAD_SHOP_WORDS)
            & ((pl.col("label_in_tradename") == 1) | pl.col("location_name_norm").str.contains(CANNABIS_NAME_WORDS)))
)
same_number = (no_store_at_address.select(STOREFRONT + ["tradenames", "house_number", "street_words"])
    .join(distinct_labels, on = "house_number", how = "inner")
    .filter(same_street("street_words", "label_street_words"))
)
same_number = (same_number
    .join(tradename_overlap(same_number.select(LABEL_ID + STOREFRONT + ["location_name_norm", "tradenames"]), LABEL_ID + STOREFRONT), on = LABEL_ID + STOREFRONT, how = "inner")
    .filter((pl.col("name_overlap") >= 0.5) | pl.col("location_name_norm").str.contains(CANNABIS_NAME_WORDS))
)
candidate_links = (pl.concat([same_city.select(LABEL_ID + STOREFRONT), same_number.select(LABEL_ID + STOREFRONT)])
    .unique()
    .join(at_storefronts.select(LABEL_ID + STOREFRONT), on = LABEL_ID + STOREFRONT, how = "anti", nulls_equal = True) # already linked by address and zip
)

different_address = (place_labels
    .drop(STOREFRONT)
    .join(candidate_links, on = LABEL_ID, how = "inner", nulls_equal = True)
    .join(storefronts, on = STOREFRONT, how = "inner")
    .join(storefront_months, on = STOREFRONT + ["month"], how = "left")
    .with_columns(pl.col("operating_licenses").is_not_null().alias("wslcb_operating"), pl.col("current_list_only").fill_null(False),
                  pl.lit("different_address").alias("address_match"))
)
at_storefronts = pl.concat([at_storefronts, check_names(different_address)], how = "diagonal_relaxed")
print("storefronts with nothing that could be the store at the exact address:", no_store_at_address.shape[0],
      "| with a candidate somewhere else:", candidate_links.select(STOREFRONT).unique().shape[0], "| candidate places:", candidate_links["safegraph_place_id"].n_unique())

# Manual verdicts (one per place, name, and storefront, from review/1e_review_page.html), if there are any yet
REVIEW_FILE = "review/1e_review_verdicts.csv"
if os.path.exists(REVIEW_FILE):
    verdicts = pl.read_csv(REVIEW_FILE, infer_schema = False).select(["safegraph_place_id", "location_name"] + STOREFRONT + ["verdict"])
    at_storefronts = (at_storefronts
        .join(verdicts, on = ["safegraph_place_id", "location_name"] + STOREFRONT, how = "left")
        .with_columns(
            pl.when((pl.col("name_check") == "review") & (pl.col("verdict") == "yes")).then(pl.lit("yes_in_review"))
              .when((pl.col("name_check") == "review") & (pl.col("verdict") == "no")).then(pl.lit("no_in_review"))
              .otherwise(pl.col("name_check"))
              .alias("name_check")
        )
        .drop("verdict")
    )
    print("verdicts read in:", verdicts.shape[0], "|", at_storefronts.group_by("name_check").agg(pl.col("safegraph_place_id").n_unique()).sort("name_check").rows())

match_rates(at_storefronts.filter(pl.col("wslcb_operating") & pl.col("name_check").is_in(NAME_OK)), "+ manual verdicts = APPROVED")
match_rates(at_storefronts.filter(pl.col("wslcb_operating") & pl.col("name_check").is_in(NAME_OK + ["review"])), "  (all review = yes)")""")

code("""# (6) Building outlines from the Oct 2026 Places & Geometry file. Two places "share" an outline when their polygons are identical. SafeGraph's own
#     POLYGON_CLASS only compares places that are open now, so closed places (a lot of the 2019-2021 stores) are checked against everything in the
#     file, open or closed (except places that closed before 2019, which can't have overlapped)
GEOMETRY_FOLDER = "raw-data/SafeGraph-Places-Geometry-2026-10"
geometry = (pl.scan_parquet(f"{GEOMETRY_FOLDER}/*.parquet")
    .select([pl.col("PLACEKEY").alias("placekey"),
             pl.col("LOCATION_NAME").alias("location_name"),
             pl.col("STREET_ADDRESS").alias("street_address"),
             pl.col("CITY").alias("city"),
             pl.col("POLYGON_CLASS").alias("polygon_class"),
             pl.col("GEOMETRY_TYPE").alias("geometry_type"),
             pl.col("PARENT_PLACEKEY").alias("parent_placekey"),
             pl.col("ENCLOSED").alias("enclosed"),
             pl.col("CLOSED_ON").alias("closed_on"),
             pl.col("POLYGON_WKT").hash().alias("outline_id")])
    .filter(pl.col("closed_on").is_null() | (pl.col("closed_on") >= pl.date(2019, 1, 1)))
    .with_columns(
        pl.when(pl.col("geometry_type") == "POLYGON").then(pl.col("outline_id")).alias("outline_id"), # points have no outline to share
        normalize_address(pl.col("street_address")).alias("street_address_norm"),
        normalize_text(pl.col("city")).alias("city_norm"),
        normalize_text(pl.col("location_name")).alias("location_name_norm"),
    )
    .collect()
)

# Everything else on the same outline as each store placekey, and whether it's just another listing of the same store
store_placekeys = (at_storefronts
    .filter(pl.col("name_check") != "no")
    .select(["placekey", "city_norm", "tradenames"])
    .drop_nulls("placekey")
    .unique("placekey")
)
outline_partners = (store_placekeys
    .join(geometry.select(["placekey", "outline_id"]).drop_nulls("outline_id"), on = "placekey", how = "inner")
    .join(geometry.select(["outline_id", pl.col("placekey").alias("partner_placekey"), pl.col("location_name").alias("partner_name"),
                           pl.col("location_name_norm").alias("partner_name_norm")]), on = "outline_id", how = "inner")
    .filter(pl.col("partner_placekey") != pl.col("placekey"))
    .explode("tradenames")
    .with_columns(pl.max_horizontal(
        name_overlap(normalize_text(pl.col("tradenames")), pl.col("partner_name_norm"), pl.col("city_norm").str.split(" ")),
        name_overlap(pl.col("partner_name_norm"), normalize_text(pl.col("tradenames")), pl.col("city_norm").str.split(" ")),
    ).alias("partner_overlap"))
    .group_by(["placekey", "partner_placekey", "partner_name", "partner_name_norm"])
    .agg(pl.col("partner_overlap").max())
    .with_columns(
        (((pl.col("partner_overlap") >= 0.5) | pl.col("partner_name_norm").str.contains(CANNABIS_NAME_WORDS))
         & ~pl.col("partner_name_norm").str.contains(HEAD_SHOP_WORDS)).alias("partner_is_the_store")
    )
    .group_by("placekey")
    .agg(
        pl.col("partner_name").filter(~pl.col("partner_is_the_store")).unique().sort().alias("other_businesses_on_outline"),
        pl.col("partner_name").filter(pl.col("partner_is_the_store")).unique().sort().alias("same_store_on_outline"),
    )
)

# For store placekeys that aren't in the 2026 file at all: what's at the same address now, as a stand-in
address_outlines = (geometry
    .with_columns((pl.len().over("outline_id") > 1).fill_null(False).alias("shares_outline"))
    .join(storefronts.select(STOREFRONT), on = STOREFRONT, how = "semi")
    .group_by(STOREFRONT)
    .agg(pl.col("shares_outline").any().alias("address_shares_outline"))
)

store_outlines = (store_placekeys.select("placekey")
    .join(geometry.select(["placekey", "geometry_type", "parent_placekey", "enclosed"]), on = "placekey", how = "left")
    .join(outline_partners, on = "placekey", how = "left")
    .with_columns(
        pl.when(pl.col("geometry_type").is_null()).then(pl.lit("not_in_2026_file"))
          .when(pl.col("geometry_type") != "POLYGON").then(pl.lit("point_only"))
          .when(pl.col("other_businesses_on_outline").list.len() > 0).then(pl.lit("shares_with_other_business"))
          .when(pl.col("same_store_on_outline").list.len() > 0).then(pl.lit("shares_with_same_store"))
          .otherwise(pl.lit("own_outline"))
          .alias("outline_status")
    )
    .drop("geometry_type")
)

at_storefronts = (at_storefronts
    .join(store_outlines, on = "placekey", how = "left")
    .join(address_outlines, on = STOREFRONT, how = "left")
    .with_columns(
        pl.when(pl.col("outline_status").is_in(["own_outline", "shares_with_same_store"])).then(pl.lit("own"))
          .when(pl.col("outline_status") == "shares_with_other_business").then(pl.lit("shared"))
          .when((pl.col("outline_status") == "not_in_2026_file") & pl.col("address_shares_outline")).then(pl.lit("likely_shared"))
          .when((pl.col("outline_status") == "not_in_2026_file") & pl.col("address_shares_outline").is_not_null()).then(pl.lit("likely_own"))
          .when(pl.col("outline_status").is_not_null() | pl.col("placekey").is_null()).then(pl.lit("unknown"))
          .alias("outline_evidence")
    )
)
store_rows = at_storefronts.filter(pl.col("name_check") != "no")
print(store_rows.group_by(["outline_evidence", "outline_status"]).agg(pl.col("safegraph_place_id").n_unique().alias("places"), pl.col("num_visits").sum()).sort(["outline_evidence", "outline_status"]))

passing = at_storefronts.filter(pl.col("wslcb_operating") & pl.col("name_check").is_in(NAME_OK))
OUTLINE_OK = ["own", "likely_own", "unknown"] # the stricter version, not part of "approved" for now
match_rates(passing.filter(pl.col("outline_evidence").is_in(OUTLINE_OK + ["likely_shared"])), "  (outline not shared)")
match_rates(passing.filter(pl.col("outline_evidence").is_in(OUTLINE_OK)), "  (or likely shared)")
match_rates(passing.filter(pl.col("outline_evidence").is_in(["own", "likely_own"])), "  (own / likely own)")
match_rates(passing.filter(pl.col("outline_evidence") == "own"), "  (own outline only)")""")

code("""# (7) Putting it together
## Write-up: section 0.2 (the approval rule and the match rates)
storefront_place_months = (at_storefronts
    .with_columns((pl.col("wslcb_operating") & pl.col("name_check").is_in(NAME_OK)).alias("approved"))
    .select(LABEL_KEY + ["placekey", "placekey_filled_in", "naics_code", "num_visits", "street_address_norm", "city_norm", "address_match", "tradenames", "operating_licenses",
                         "current_list_only", "name_overlap", "name_rule", "name_check", "wslcb_operating", "outline_evidence", "outline_status",
                         "other_businesses_on_outline", "parent_placekey", "enclosed", "approved"])
    .sort(["street_address_norm", "city_norm", "safegraph_place_id", "month", "location_name", "street_address", "city"], nulls_last = True)
)

storefront_place_months.write_parquet("processed-data/1e_storefront_place_months.parquet")

print("[1e] rows:", storefront_place_months.shape[0], "| approved:", storefront_place_months["approved"].sum(),
      "| approved visits:", storefront_place_months.filter(pl.col("approved"))["num_visits"].sum(),
      "| approved places:", storefront_place_months.filter(pl.col("approved"))["safegraph_place_id"].n_unique())
pl.DataFrame(match_rate_log)""")

code("""### IRREGULARITIES, PART E
# (1) Blank placekeys in Veraset, by month (the visits still have a safegraph_place_id, which is why that's the key here)
print(veraset_pois.group_by("month").agg(pl.col("num_visits").filter(pl.col("placekey").is_null()).sum().alias("visits_without_placekey"), pl.col("num_visits").sum())
      .filter(pl.col("visits_without_placekey") > 0).sort("month").rows())
print("store labels (not 'no') with a filled-in placekey:", storefront_place_months.filter((pl.col("name_check") != "no") & pl.col("placekey_filled_in"))["num_visits"].sum(), "visits",
      "| with no placekey in any month:", storefront_place_months.filter((pl.col("name_check") != "no") & pl.col("placekey").is_null())["num_visits"].sum(), "visits")

# (2) safegraph_place_id and placekey aren't one-to-one
id_pairs = veraset_pois.filter(pl.col("placekey").is_not_null()).select(["safegraph_place_id", "placekey"]).unique()
print("place IDs with 2+ placekeys:", id_pairs.group_by("safegraph_place_id").len().filter(pl.col("len") > 1).shape[0], "of", id_pairs["safegraph_place_id"].n_unique(),
      "| placekeys with 2+ place IDs:", id_pairs.group_by("placekey").len().filter(pl.col("len") > 1).shape[0], "of", id_pairs["placekey"].n_unique())

# (3) Store names showing up in months when WSLCB doesn't have the store operating (Safegraph keeping a closed store's name around, or a store
#     dropped from the WSLCB lists early)
outside = storefront_place_months.filter(pl.col("name_check").is_in(NAME_OK) & ~pl.col("wslcb_operating"))
print("approved-looking labels outside the WSLCB window:", outside["safegraph_place_id"].n_unique(), "places,", outside["num_visits"].sum(), "visits")

# (4) Outlines: Safegraph's POLYGON_CLASS against the identical-polygon check used here (they differ when a closed place shares the outline)
print(store_placekeys.select("placekey").join(geometry.select(["placekey", "polygon_class"]), on = "placekey", how = "inner")
      .join(store_outlines.select(["placekey", "outline_status"]), on = "placekey", how = "left")
      .group_by(["polygon_class", "outline_status"]).len().sort(["polygon_class", "outline_status"]).rows())

# (5) Filtering the raw visits with [1e]: a month with lots of blank placekeys, joined on the label key, should give back exactly the approved visits
CHECK_MONTH = "2020-12"
approved_labels = storefront_place_months.filter(pl.col("approved") & (pl.col("month") == CHECK_MONTH)).select(LABEL_KEY[1:]).unique()
approved_visits = (pl.scan_parquet(f"raw-data/veraset-visits-wa/month={CHECK_MONTH}/*.parquet")
    .join(approved_labels.lazy(), on = LABEL_KEY[1:], how = "semi", nulls_equal = True)
    .select(pl.len(), pl.col("placekey").is_null().sum().alias("without_placekey"))
    .collect(engine = "streaming")
)
print(CHECK_MONTH, "visits through the join:", approved_visits.rows(), "| approved visits in [1e]:",
      storefront_place_months.filter(pl.col("approved") & (pl.col("month") == CHECK_MONTH))["num_visits"].sum())

# (6) The biggest review items, for a sense of what's waiting
(storefront_place_months
    .filter((pl.col("name_check") == "review") & pl.col("wslcb_operating"))
    .group_by(["safegraph_place_id", "location_name", "street_address", "street_address_norm", "city_norm", "address_match", "name_rule"])
    .agg(pl.col("tradenames").first(), pl.col("num_visits").sum())
    .sort("num_visits", descending = True)
    .head(20)
)""")

md("""### IRREGULARITIES LOG 
(Written from the outputs above as of the last run. If the inputs change, re-check the numbers.) 

**WSLCB samples** 
- File names don't match the dates on the sheets inside them. `MarijuanaApplicants - JANUARY 2020.xls` is actually the 11-3-20 sample, and `MarijuanaApplicants - NOVEMBER.xls` (called "NOVEMBER 2019" in the old folder) is the 1-28-20 sample. Sheet dates are used everywhere here. 
- That leaves samples at 2018-07-17, 2019-01-18, 2020-01-28, 2020-11-03, and 2021-04-13: nothing for the whole of 2019 after January, and nothing after Apr 2021 except the undated current list. The current list says "CANNABIS RETAILER" instead of "MARIJUANA RETAILER", so it's from after WA's 2022 renaming. 
- Headers aren't consistent. The Jan 2019 retailer sheet's first column is called "." instead of "Tradename", and the Jul 2018 file has "DateCreated" where the others have "DateIssued". 
- "DateIssued" isn't a fixed issue date: it changes between samples for 696 of the 826 licenses that have one, so it's more like a last-renewal/last-action date. 
- These are *applicant* lists. "PENDING (NOT ISSUED)" rows are applications (often a store's pending move to a new address), not stores. Only "ACTIVE (ISSUED)", "PENDING (ISSUED)", and "CLOSED (TEMPORARY)" count as operating; 335 of the 1,048 license-address pairs in [1a] never reach one of those. 
- Later samples quietly drop old licenses: 227 licenses in the Jan 2019 sample are simply absent from the Jan 2020 one, and 24 of those were operating in Jan 2019 with no "CLOSED" row ever showing up. Disappearing from the list isn't the same as a recorded closure. 
- Title certificate holders have "RETAIL TITLE CERTIFICATE" instead of a street address (58 license-address pairs), and some tradenames carry "(TITLE OWNER ONLY)". 
- Licenses listed twice in the same sample: 27 in Jul 2018 (a current address plus a pending move, or the same store under an old and a new name, typos like "NORHTWEST CANNABIS" included) and 283 in the current list, which keeps history rows. 
- Over the samples, 78 licenses show up at 2+ addresses, 264 under 2+ tradenames, and 267 under 2+ UBIs (ownership changes). 115 addresses have 2+ license numbers on them, eg. a pending license number replaced by a new one once issued, or a new owner getting a new license. 
- The current list has at least one row that's clearly a different business under a cannabis license number (license 79720, "PHO HOUSE & GRILL"). 
- 90 licenses appear only in the dated samples, and 98 appear only in the current list (opened after Apr 2021). 

**Matching to Safegraph** 
- The Safegraph file is missing 30,746 of the 115,019 placekeys that show up in Veraset, including about 120 that are obviously cannabis stores by name. Veraset's own POI info for those placekeys was added as a second source, and 198 of the 728 [1b] placekeys come from it. 
- Safegraph files most dispensaries under NAICS 424210 ("Drugs and Druggists' Sundries Merchant Wholesalers"). A few show up under unrelated codes (eg. "Local Roots Marijuana" as a health practitioner, "CannaZone" as a grocery) or truncated codes ("44"). 
- Matching by address picks up every business at that address (up to 60+ in a strip mall), hence "match_type". "name", "cannabis_name", and "category" matches look right on spot checks. Most "related" matches (tobacco/misc. stores with no name overlap) are vape shops next door, not the dispensary, so they're dropped wherever the same location has a better match. The 21 that are left are the only thing found for their location, and most of them are still probably the smoke shop next door. 
- Addresses are written inconsistently enough to break exact matching: "SR 20" / "STATE ROUTE 20" / "State Highway 20" / "WA-20", "FOURTH PLAIN" / "4th Plain", unit numbers stuck on the end ("939 N CALLOW AVE 100B"), Ave vs. Way vs. Rd, postal city vs. actual city (Lake Forest Park vs. Shoreline, Millwood vs. Spokane Valley). The normalizer now handles the first three, and fallback passes 5-7 catch the rest (70 [1b] rows). 
- Name overlap used to count generic and city words, which matched eg. "ANACORTES CANNABIS" to "Anacortes Flea Market" and "FIFE CANNABIS" to "Fife Pharmacy". Those words don't count anymore. 
- "POT" and "WEED" in the cannabis-name pattern also catch "Top Pot Doughnuts" and "Benton County Noxious Weed Brd", so cannabis_name matches without a name match aren't bulletproof either. 
- Lots of stores were renamed after the WSLCB samples (eg. THE REEF -> "Ruckus Recreational Fremont 21"), which is why the cannabis-name and category rules matter. 
- 643 of 931 licenses get at least one cannabis placekey, and 526 of those have Veraset visits (week 7: 400). 121 of the 702 operating license-address pairs still have no match. 51 cannabis-looking places (by name and category) don't match any WSLCB location. 
- 138 placekeys are claimed by 2+ licenses: the same storefront under successive licenses/owners. 

**what@where (Part B')** 
- Placekeys do NOT reliably change when the business changes. Across Veraset, 1,073 placekeys switch to a name sharing no words with the old one, eg. "Zoostore Woodland Park Zoo" -> "Molbak's Butterfly Garden", "Good Guys Foodmart" -> "PFI Mart", "GENESIS OF EVERETT" -> "Lee Johnson Hyundai Of Everett". Some of these are rebrands of the same business, but plenty aren't. 
- Among the [1b] placekeys, the clearest case is `224-222@5x2-zdz-6ff`, which is "Baker Bear Grocery" (a grocery, NAICS 445110) in Veraset up to Aug 2020 and "2020 Solutions" (424210) by Sep 2021, keeping the same placekey. That matches the WSLCB record of the "2020 CANNABIS SOLUTIONS MT BAKER HIGHWAY" license going from pending in 2020 to active in 2021. Visits to that placekey before late 2020 are grocery visits. 
- The other cannabis name changes are cannabis-to-cannabis rebrands or formatting ("THC of Lacey" <-> "Bud Stop", "Kush and Glass" -> "Kush21 Burien's First Pot Shop", "KushKlub" <-> "Kush Klub"). 
- NAICS codes in Veraset are noisy and can't be used to spot a change of business. 238 of the 492 [1b] placekeys seen in Veraset change NAICS code, mostly bouncing between unrelated codes with the same name: 424210 <-> 446110 (pharmacies), 722515 (coffee shops) -> 424210, 448120 (clothing) -> 722515. The changes bunch up in a few months (2020-08, 2019-12, 2020-02, 2021-06), which look like Safegraph data releases. 
- Safegraph's open/close dates are too sparse to check this from the other direction (opened_on is filled in for 57 of 5,997 placekeys sharing a "where" with [1b]), so no replacement businesses can be confirmed that way. 
- Takeaway: when counting dispensary visits, restrict each placekey to the months when it's actually a cannabis store. Use the WSLCB operating window from [1a] and/or a cannabis-looking name in Veraset that month, rather than trusting the placekey alone. 

**Geography (Part C)** 
- 530 of the 728 [1b] placekeys have Safegraph coordinates. The other 198 (Veraset-only) are placed using a point inside their Veraset 2010 CBG, which is fine for CBG/county but rougher for place and ZCTA. 
- Sources mostly agree: the point's 2010 CBG matches Veraset's CBG for 288 of 294, the point's ZCTA matches Veraset's zip code for 280 of 294, and it matches the WSLCB zip for 497 of 530 (ZCTAs aren't zip codes, so some mismatch is expected). 
- The census county matches the WSLCB county for 727 of 728. The one miss is a WSLCB error: license 422779 ("KITTITAS CANNABIS", 36711 Hwy 12, Dayton) is listed under Kittitas County, but Dayton is in Columbia County. 
- 115 placekeys sit outside any census place (unincorporated, non-CDP areas). Of the rest, 471 are in cities, 140 in CDPs, and 2 in places the census calls a "town" (Twisp and Winthrop). 

**Coordinates (Part D)** 
- The Census geocoder finds 852 of the 990 addresses (517 exact, 335 non-exact). It misses mostly state-highway and newer-road addresses. 
- Where both exist, the Census point is a median 54 m from Safegraph's point for the same store (90th percentile 182 m, 28 over 250 m). Non-exact matches sometimes land on the wrong street (E vs W), and rural highway addresses get interpolated along long blocks, so Safegraph's point goes first. 
- Final sources: 737 same-address Safegraph POI, 108 Census exact, 104 Census non-exact, 13 Veraset placekey hexagon, 8 matched placekey, 7 OpenStreetMap, 13 ZCTA. 
- Census can't read "½" (it comes back garbled as "930Â½ VALLEY MALL PKWY" and lands on "2 VALLEY MALL PKWY"). 

**Store visits without distances (Part E)** 
- 546 of the 931 licenses were operating in some month of 2019-2021 by the dated samples, at 542 storefronts (17,742 storefront-months). Another 94 licenses are only operating in the undated current list, so they might have opened after 2021 and don't count in the denominators. 
- Veraset leaves the placekey blank on 1.7M visits, from Nov 2020 (328k that month) through Jul 2021 (85k), and a few thousand a month after that. Every visit still has `safegraph_place_id`, so that's the key. 36,248 visits to store-looking labels only make it in through that (the placekey gets filled in from other months), and 9,640 have no placekey in any month. Filtering on placekey alone would drop all of them. 
- `safegraph_place_id` and placekey aren't one-to-one: 1,982 place IDs have 2+ placekeys over the three years, and 2,070 placekeys have 2+ place IDs. 
- Veraset often names the unincorporated area instead of the postal city: "tanglewilde" for Olympia/Lacey, "parkland" for Tacoma, "summit view" and "clover creek" for Puyallup, "west richland" for Richland, "otis orchards-east farms", "north lynnwood", "millwood" for Spokane. Requiring the same city dropped real stores (THC of Lacey, Primo Cannabis, The Gallery Parkland, Sweet Leaf), so the same street address plus the same zip counts too (1,765 labels). 
- Match rates (licenses / storefronts operating in 2019-2021): 78.4% / 77.5% have any Veraset place at the WSLCB address. 58.2% / 57.7% have one whose name is automatically accepted as the store. If every review item turned out to be the store, the ceiling is 80.4% / 79.5%. With the review verdicts (241 yes, 17 no), 79.7% / 78.8% are approved (1,063,478 visits). Dropping places whose building outline is shared with another business would take that to 71.1% / 70.1% (916,209 visits), and also dropping the ones that are likely shared to 62.3% / 61.4% (802,871 visits). With only outlines known to be the store's own, 40.7% / 39.9%. The other ~20% of storefronts have nothing in Veraset that passes even the looser review rules. 
- Store names at the right address that are something else: "Origins Coffee Roasters" at ORIGINS, "Gypsy Glass" at GYPSY GREEN, "KushMart Clothing & Glass" at KUSHMART, "The Market at Birch Bay" at BIRCH BAY BUDZ. This is why only labels with nothing but tradename/cannabis words in them are accepted without review. 
- The different-address candidates are a mix of address spelling (420 CARPENTER at "422 carpenter rd se ste 105", Have a Heart Skyway, where WSLCB's address stops at "MARTIN LUTHER KING JR" with no "WAY S", Apex at "gilbert ave" vs "GILBERT RD"), postal city vs. actual city (Sativa Sisters is in "spokane valley" / "millwood" in Veraset and SPOKANE in WSLCB) and unrelated namesakes (Homegrown the sandwich chain for HOMEGROWN NORTHWEST). 
- 53 places (41,265 visits) carry the store's name in months when no license is listed as operating there. 15,690 of those visits count anyway under another spelling of the same storefront ("12833 MARTIN LUTHER KING JR WA", "E GILBERT AVE" vs "RD", "SUNSET BLVD N" vs "NE"). About 4,600 are WSLCB listing the store a few doors off (Nimbin at 10859 vs 10861 1st Ave S, Sweet Green NW at 19941 vs 19943 Old Hwy 99, "120 NE CANNING DR" vs "120 CANNING DR NE"). The rest are names Veraset keeps for months or years after WSLCB has the store closed (The Pot Shop in Bellingham, King Cronic III in Longview). These aren't approved. 
- Building outlines: 74 store places share an outline with a different business (204k visits), eg. Main Street Marijuana East with physical therapists, Cinder Spokane Valley with a cafe and a rehab clinic, Sativa Sisters with a church, Fweedom Mountlake Terrace with 32 offices. Another 81 aren't in the 2026 file and their address has a shared outline now (134k visits). Since Veraset may split the visits in a shared outline between the businesses in it, a store's count there might include some of the neighbors' visitors (or miss some of its own), so `outline_evidence` is in [1e] for checking results without them. 
- Safegraph's own POLYGON_CLASS only compares places that are open now. 11 OWNED_POLYGON store placekeys have exactly the same outline as a place that has since closed, and 17 of the 35 closed store placekeys share theirs. Those count as shared here. 
- The outlines are from Oct 2026, not 2019-2021. A building split or merged since then would be wrong here. 
- The Oct 2026 file isn't a newer copy of the July file used in Part B: 18,493 of the July file's 568,297 placekeys aren't in it. 
- Checked: filtering Dec 2020's raw visits by joining on the label key gives back exactly the approved count in [1e] (26,352 visits, 4,028 of them with no placekey). 
""")

code("")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, sys.argv[1])
