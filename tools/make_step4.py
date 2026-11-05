import nbformat as nbf
import sys

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""This document aims to accomplish the following:

(a) Give every sample device a home city, and a crosswalk from devices to their home city/county/zip/CBG

(i) Match each device from [2ai] to a home city using [3d]. Combine with known information on CBG/zip to create a crosswalk between city/county/zip/CBG and devices. Label as ([4ai])

(ii) Match each device from [2aii] to a home city using [3d]. Combine with known information on CBG/zip to create a crosswalk between city/county/zip/CBG and devices. Label as ([4aii])

(b) Create intermediate tables of visits to dispensaries (the approved store labels in [1e]), for

(i) the year-restricted data. Label as ([4bi])

(ii) the quarter-restricted data. Label as ([4bii])

(c) In the same style as the week 6 notebook, recreate the charts for demographic representativeness, dispensary visit distributions, and visit start times/durations. Split by the yearly and quarterly samples. Alongside the charts, the numbers behind them:

(i) sample sizes by year/quarter, and how well the home counties line up with the 2020 census. Label as ([4ci]), with the county by county comparison as ([4ci'])

(ii) the share of devices with at least one dispensary visit, and among those, the distribution of visits (over the window and per 30 observed days) and of distinct stores (licenses) visited. Label as ([4cii])

(iii) visit start times by day of week and hour. Label as ([4ciii]). The share of visits outside 8 am-midnight and the distribution of minimum dwell times. Label as ([4ciii'])

Notes:
- Veraset only gives a home as a 2010 CBG. CBGs don't nest in cities, so the home city is the census place (city, town, or CDP) holding the biggest share of the CBG's 21+ population, worked out block by block with the Census 2020 block assignment file. That share is kept, so devices in CBGs split by a city line can be dropped or weighted later
- A dispensary visit is a visit Veraset attributed to an approved [1e] label (place ID + name + address + city, that month). No distances anywhere
- Visits get cleaned before anything is counted (see Part B): duplicates dropped, visits Veraset cut at UTC midnight put back together, and only visits inside the device's window kept: the calendar year in local time for the yearly sample, and the device's 60+ day stretch (cut to the local calendar quarter) for the quarterly sample
- Charts also get saved to "Charts and Figures" as .html files
""")

code("""### IMPORTS AND PRELIMINARIES
## MEMORY STUFF
# Stops polars from holding onto freed memory, and forces it to be returned to the OS
import os
os.environ['_RJEM_MALLOC_CONF'] = "background_thread:true,dirty_decay_ms:0,muzzy_decay_ms:0"

## WORKHORSE LIBRARY
# Enables data streaming of large files
import polars as pl

## CHARTS (same as week 6)
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2 as chi2_distribution # p-values for the representativeness test

## EXTRAS
# For grabbing the Census block assignment file
import zipfile
import urllib.request
import datetime # for the start of each window, as a timestamp

# Cleans up some cell outputs
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

STATE_FIPS = "53" # Washington
YEARS = ["2019", "2020", "2021"]
QUARTERS = [1, 2, 3, 4]""")

code("""### PART A, INTERMEDIATE WORK: WHICH CENSUS PLACES EACH 2010 CBG FALLS IN
## Write-up: section 0.4 (home city)
## Output: one row per 2010 CBG x census place (or no place at all) that it overlaps, with the share of the CBG's 21+ population living there
##         Columns are "cbg2010", "place_fips" (null for the part outside any city/town/CDP), "population_total", "population_21plus" (floats, blocks get
##         split by area the same way as [3b']), "share_21plus"
## Name: cbg2010_place_shares.parquet
# Method: the Census 2020 block assignment file says which place every 2020 block is in (blocks nest in places, CBGs don't). Each 2020 block's population
#         gets split across the 2010 blocks it overlaps by land area (exactly like [3b'] in Step 3), and summed up by 2010 CBG and place
BAF_PATH = "raw-data/census/BlockAssign_ST53_WA.zip"
REL_PATH = "raw-data/census/TAB2010_TAB2020_ST53.zip"
BLOCKS_PATH = "raw-data/census/p12_blocks_2020.parquet" # cached by Step 3

if not os.path.exists(BAF_PATH):
    urllib.request.urlretrieve("https://www2.census.gov/geo/docs/maps-data/data/baf2020/BlockAssign_ST53_WA.zip", BAF_PATH)

with zipfile.ZipFile(BAF_PATH) as z:
    block_places = pl.read_csv(z.read("BlockAssign_ST53_WA_INCPLACE_CDP.txt"), separator = "|", infer_schema = False)
block_places = block_places.select([
    pl.col("BLOCKID").alias("block2020"),
    (STATE_FIPS + pl.col("PLACEFP")).alias("place_fips"), # stays null for blocks outside every place
])

with zipfile.ZipFile(REL_PATH) as z:
    rel_df = pl.read_csv(z.read(z.namelist()[0]), separator = "|", infer_schema = False)
rel_df = rel_df.rename({rel_df.columns[0]: "STATE_2010"}) # first column name comes with a byte-order mark stuck on the front

rel_df = (rel_df
    .with_columns(
        pl.col(["AREALAND_2020", "AREAWATER_2020", "AREALAND_INT", "AREAWATER_INT"]).cast(pl.Int64),
        (pl.col("STATE_2020") + pl.col("COUNTY_2020") + pl.col("TRACT_2020") + pl.col("BLK_2020")).alias("block2020"),
        (pl.col("STATE_2010") + pl.col("COUNTY_2010") + pl.col("TRACT_2010") + pl.col("BLK_2010").str.slice(0, 1)).alias("cbg2010"),
    )
    .filter(pl.col("STATE_2020") == STATE_FIPS)
    .with_columns( # share of the 2020 block falling in each piece. Land area if the block has any, water area if it's all water, an even split if neither
        weight = pl.when(pl.col("AREALAND_2020") > 0).then(pl.col("AREALAND_INT") / pl.col("AREALAND_2020"))
                   .when(pl.col("AREAWATER_2020") > 0).then(pl.col("AREAWATER_INT") / pl.col("AREAWATER_2020"))
                   .otherwise(1 / pl.len().over("block2020"))
    )
)

blocks_df = pl.read_parquet(BLOCKS_PATH)
print("2020 blocks without a place assignment row:", blocks_df.join(block_places, on = "block2020", how = "anti").shape[0])

cbg_places = (blocks_df
    .join(block_places, on = "block2020", how = "left")
    .join(rel_df.select(["block2020", "cbg2010", "STATE_2010", "weight"]), on = "block2020", how = "left")
    .filter(pl.col("STATE_2010") == STATE_FIPS)
    .group_by(["cbg2010", "place_fips"])
    .agg(
        (pl.col("population_total") * pl.col("weight")).sum().alias("population_total"),
        (pl.col("population_21plus") * pl.col("weight")).sum().alias("population_21plus"),
        pl.col("weight").sum().alias("num_blocks"), # (fractional) number of 2020 blocks, a stand-in for CBGs with nobody 21+ in them
    )
    .with_columns(
        pl.when(pl.col("population_21plus").sum().over("cbg2010") > 0)
          .then(pl.col("population_21plus") / pl.col("population_21plus").sum().over("cbg2010"))
          .otherwise(pl.col("num_blocks") / pl.col("num_blocks").sum().over("cbg2010"))
          .alias("share_21plus")
    )
    .drop("num_blocks")
    .sort(["cbg2010", "place_fips"], nulls_last = True)
)

cbg_places.write_parquet("processed-data/cbg2010_place_shares.parquet")

# One place per CBG: whichever holds the biggest share of its 21+ population (which might be "no place" for rural CBGs)
cbg_home_place = (cbg_places
    .sort(["cbg2010", "share_21plus", "place_fips"], descending = [False, True, False], nulls_last = True)
    .unique("cbg2010", keep = "first", maintain_order = True)
    .select([pl.col("cbg2010").alias("home_cbg"), pl.col("place_fips").alias("home_place_fips"), pl.col("share_21plus").alias("home_place_share")])
)

print("2010 CBGs:", cbg_home_place.shape[0], "| entirely in one place (or none):", cbg_home_place.filter(pl.col("home_place_share") > 0.999).shape[0],
      "| biggest place under half:", cbg_home_place.filter(pl.col("home_place_share") < 0.5).shape[0],
      "| mostly outside any place:", cbg_home_place.filter(pl.col("home_place_fips").is_null()).shape[0])""")

code("""### PART A, HELPER
## Takes a home location table from [2b] and adds the home county name, home city (place), and whether the home zip is a WA ZCTA in [3a]
# Columns are "caid_hash", "home_cbg" (2010 CBG), "home_zip_code", "home_zip_in_3a", "home_county_fips", "home_county_name", "home_place_fips",
# "home_place_name", "home_place_type" (city, town, CDP, or null for no place), "home_place_share" (share of the home CBG's 21+ population in that place),
# "home_source" ("in_window" / "carried_forward", from [2b]), "home_visit_timestamp"
counties = pl.read_parquet("processed-data/3c_county_population_2020.parquet").select([pl.col("county_fips").alias("home_county_fips"), pl.col("county_name").alias("home_county_name")])
places = pl.read_parquet("processed-data/3d_place_population_2020.parquet").select([pl.col("place_fips").alias("home_place_fips"), pl.col("place_name").alias("home_place_name"),
                                                                                    pl.col("place_type").alias("home_place_type")])
zctas = pl.read_parquet("processed-data/3a_zcta_population_2020.parquet").select(pl.col("zcta").alias("home_zip_code"))

def home_geography(homes):
    return (homes
        .join(cbg_home_place, on = "home_cbg", how = "left") # homes outside WA get nulls here
        .join(places, on = "home_place_fips", how = "left")
        .join(counties, on = "home_county_fips", how = "left")
        .with_columns(pl.col("home_zip_code").is_in(zctas["home_zip_code"].implode()).alias("home_zip_in_3a"))
        .select(["caid_hash", "home_cbg", "home_zip_code", "home_zip_in_3a", "home_county_fips", "home_county_name", "home_place_fips", "home_place_name",
                 "home_place_type", "home_place_share", "home_source", "home_visit_timestamp"])
    )""")

code("""### PART A, SUBPART I
## Write-up: sections 0.4 and 4 (home geography, yearly sample)
## Output: one row per [2ai] device (350+ days that year) per year, with its home CBG, zip, county, and city
##         Columns are "year" plus the helper's columns. Devices [2b] couldn't find any home for are left out (see [2biii])
## Name: 4ai_device_home_geography_yearly.parquet
results = []
for YEAR in YEARS:
    homes = pl.read_parquet(f"processed-data/home_locations_yearly_{YEAR}.parquet")
    num_devices = pl.scan_parquet(f"processed-data/devices_with_350days_{YEAR}.parquet").select(pl.len()).collect().item()
    results.append(home_geography(homes).with_columns(pl.lit(int(YEAR)).cast(pl.Int32).alias("year")))
    print(YEAR, "| [2ai] devices:", num_devices, "| with a home:", homes.shape[0])

yearly_crosswalk = pl.concat(results).select(["year"] + results[0].columns[:-1])
yearly_crosswalk.write_parquet("processed-data/4ai_device_home_geography_yearly.parquet")

yearly_crosswalk.group_by("year").agg(
    pl.len().alias("devices"),
    pl.col("home_county_fips").str.starts_with(STATE_FIPS).mean().alias("home_in_wa"),
    pl.col("home_place_fips").is_not_null().mean().alias("home_in_a_place"),
    (pl.col("home_place_share") < 0.5).mean().alias("place_share_under_half"),
    pl.col("home_zip_in_3a").mean(),
).sort("year")""")

code("""### PART A, SUBPART II
## Write-up: sections 0.4 and 4 (home geography, quarterly sample)
## Output: one row per [2aii] device (60+ consecutive days that quarter) per quarter, with its home CBG, zip, county, and city
##         Columns are "quarter" ("2019Q1" etc.) plus the helper's columns
## Name: 4aii_device_home_geography_quarterly.parquet
results = []
for YEAR in YEARS:
    for QUARTER in QUARTERS:
        homes = pl.read_parquet(f"processed-data/home_locations_quarterly_{YEAR}Q{QUARTER}.parquet")
        results.append(home_geography(homes).with_columns(pl.lit(f"{YEAR}Q{QUARTER}").alias("quarter")))

quarterly_crosswalk = pl.concat(results).select(["quarter"] + results[0].columns[:-1])
quarterly_crosswalk.write_parquet("processed-data/4aii_device_home_geography_quarterly.parquet")

quarterly_crosswalk.group_by("quarter").agg(
    pl.len().alias("devices"),
    pl.col("home_county_fips").str.starts_with(STATE_FIPS).mean().alias("home_in_wa"),
    pl.col("home_place_fips").is_not_null().mean().alias("home_in_a_place"),
    (pl.col("home_place_share") < 0.5).mean().alias("place_share_under_half"),
    pl.col("home_zip_in_3a").mean(),
).sort("quarter")""")

code("""### PART B, INTERMEDIATE WORK: EACH DEVICE'S WINDOW AND OBSERVED DAYS
## Write-up: section 0.3, fix (3), and "per 30 observed days" in sections 1-2
## Output: one row per sample device per year/quarter, with the days its visits get counted over (days since 1970-01-01 in local time, both ends included)
##         and how many of them it was seen on. The window is the calendar year for the yearly sample, and the device's 60+ day stretch (its longest run of
##         consecutive days, as in Step 2) for the quarterly sample, so a device in [2aii] only gets its visits during that stretch counted
##         Columns are "year" / "quarter", "caid_hash", "window_start_day", "window_end_day", "observed_days" (distinct days in the window with any record,
##         home visits included)
## Name: device_windows_yearly.parquet, device_windows_quarterly.parquet
# The sample tables are cut on UTC month folders, and each folder runs through the 1st of the next month (UTC). So a table starts ~8 hours early and ends
# ~16 hours late in local time, and Step 2 counted those days too. Here the windows are cut back to the local calendar. Devices stay in the samples as given
def day_number(year, month): # local midnight starting that month, as days since 1970-01-01
    return (datetime.date(year, month, 1) - datetime.date(1970, 1, 1)).days

def observed_days(sample_path, window_start_day, window_end_day):
    return (pl.scan_parquet(sample_path)
        .filter((pl.col("day") >= window_start_day) & (pl.col("day") <= window_end_day))
        .group_by("caid_hash")
        .agg(pl.col("day").n_unique().cast(pl.Int64).alias("observed_days"))
        .with_columns(pl.lit(window_start_day).alias("window_start_day"), pl.lit(window_end_day).alias("window_end_day"))
        .collect(engine = "streaming")
    )

results = []
for YEAR in YEARS:
    results.append(observed_days(f"processed-data/sample_restricted_yearly_{YEAR}.parquet", day_number(int(YEAR), 1), day_number(int(YEAR) + 1, 1) - 1)
        .with_columns(pl.lit(int(YEAR)).cast(pl.Int32).alias("year")))
yearly_windows = pl.concat(results).select(["year", "caid_hash", "window_start_day", "window_end_day", "observed_days"])
yearly_windows.write_parquet("processed-data/device_windows_yearly.parquet")

def longest_stretch(sample_path, quarter_start, quarter_end): # the gaps-and-islands code from Step 2, keeping the run's first and last day, then cut to the local quarter
    return (pl.scan_parquet(sample_path)
        .select(["caid_hash", "day"])
        .unique()
        .sort(["caid_hash", "day"])
        .with_columns(run = pl.col("day").diff().over("caid_hash").ne(1).fill_null(True).cast(pl.UInt32).cum_sum().over("caid_hash"))
        .group_by(["caid_hash", "run"])
        .agg(pl.col("day").min().alias("window_start_day"), pl.col("day").max().alias("window_end_day"))
        .group_by("caid_hash")
        .agg(pl.all().sort_by(pl.col("window_end_day") - pl.col("window_start_day"), descending = True).first()) # at most one run of 60+ days fits in a quarter
        .with_columns(
            pl.col("window_start_day").clip(lower_bound = quarter_start),
            pl.col("window_end_day").clip(upper_bound = quarter_end),
        )
        .with_columns((pl.col("window_end_day") - pl.col("window_start_day") + 1).cast(pl.Int64).alias("observed_days")) # every day in a run is observed
        .collect(engine = "streaming")
    )

results = []
for YEAR in YEARS:
    for QUARTER in QUARTERS:
        quarter_start = day_number(int(YEAR), 3 * QUARTER - 2)
        quarter_end = day_number(int(YEAR) + (QUARTER == 4), 3 * QUARTER % 12 + 1) - 1
        results.append(longest_stretch(f"processed-data/sample_restricted_quarterly_{YEAR}Q{QUARTER}.parquet", quarter_start, quarter_end)
            .with_columns(pl.lit(f"{YEAR}Q{QUARTER}").alias("quarter")))
quarterly_windows = pl.concat(results).select(["quarter", "caid_hash", "window_start_day", "window_end_day", "observed_days"])
quarterly_windows.write_parquet("processed-data/device_windows_quarterly.parquet")

print("yearly device-years:", yearly_windows.shape[0], "| observed days under 350 once cut to the local year:", (yearly_windows["observed_days"] < 350).sum())
print("quarterly device-quarters:", quarterly_windows.shape[0], "| stretch under 60 days once cut to the local quarter:", (quarterly_windows["observed_days"] < 60).sum(),
      "| mean stretch:", round(quarterly_windows["observed_days"].mean(), 1), "days")""")

code("""### PART B, HELPER
## Write-up: sections 0.2 (which visits count) and 0.3 (the cleaning fixes, see fix_visits below)
## The approved store labels from [1e], plus where the store is (from [1d]). Join visits to these on the label key to get dispensary visits
# A handful of labels sit at two [1e] storefronts that are really one store listed two ways by WSLCB, so each label-month keeps just one
# (exact address first), otherwise those visits would get counted twice
LABEL_KEY = ["month", "safegraph_place_id", "location_name", "street_address", "city"]
STOREFRONT = ["street_address_norm", "city_norm"]

store_labels = (pl.read_parquet("processed-data/1e_storefront_place_months.parquet")
    .filter(pl.col("approved"))
    .with_columns(
        pl.col("month").str.replace("-", "").cast(pl.Int32), # "YYYY-MM" -> int(YYYYMM), like the sample tables
        pl.col("address_match").replace_strict({"exact": 0, "same_address_and_zip": 1, "different_address": 2}).alias("address_rank"),
    )
    .sort(LABEL_KEY + ["address_rank"] + STOREFRONT, nulls_last = True)
    .unique(LABEL_KEY, keep = "first", maintain_order = True)
    .select(LABEL_KEY + STOREFRONT + ["tradenames", "operating_licenses", "address_match", "name_check", "outline_evidence"])
)

store_geography = (pl.read_parquet("processed-data/1d_wslcb_location_coordinates.parquet")
    .sort("license")
    .group_by(STOREFRONT, maintain_order = True)
    .agg(pl.col(["cbg2010", "county_fips", "county_name", "place_fips", "place_name", "place_type", "zcta"]).first())
    .rename({"cbg2010": "store_cbg", "county_fips": "store_county_fips", "county_name": "store_county_name", "place_fips": "store_place_fips",
             "place_name": "store_place_name", "place_type": "store_place_type", "zcta": "store_zcta"})
)
store_labels = store_labels.join(store_geography, on = STOREFRONT, how = "left")

print("approved label-months:", store_labels.shape[0], "| storefronts:", store_labels.select(STOREFRONT).unique().shape[0],
      "| without store geography:", store_labels.filter(pl.col("store_county_fips").is_null()).shape[0])

VISIT_COLUMNS = ["caid_hash", "utc_timestamp", "local_timestamp", "day", "week", "month", "minimum_dwell", "safegraph_place_id", "placekey", "location_name",
                 "street_address", "city", "zip_code", "census_block_group"]

def dispensary_visits(sample_path):
    return (pl.scan_parquet(sample_path)
        .select(VISIT_COLUMNS)
        .join(store_labels.lazy(), on = LABEL_KEY, how = "inner", nulls_equal = True) # nulls_equal, since street_address/city can be blank
        .collect(engine = "streaming")
    )

## Cleans one sample's dispensary visits (all its periods at once). Adds "visit_parts" (how many Veraset rows went into the visit) and
## "utc_midnight_start" (the visit still starts in the first MIDNIGHT_WINDOW of a UTC day, so it may really have started earlier)
MIDNIGHT_WINDOW = 1800 # seconds. Veraset picks a cut visit back up with the device's first ping after UTC midnight, which can take ~20-30 minutes
MAX_SPLIT_GAP = 3600 # seconds. How far apart the two halves of a visit cut at UTC midnight can be and still get put back together

def fix_visits(visits, period, windows):
    num_rows = visits.shape[0]

    # (1) The same visit can show up twice: once in each month folder for visits on the 1st (UTC), and now and then twice in one folder. Keep one
    #     (the longest, for the handful where the dwell differs)
    visits = (visits
        .sort("minimum_dwell", descending = True)
        .unique([period, "caid_hash", "utc_timestamp", "safegraph_place_id"], keep = "first")
        .sort([period, "caid_hash", "utc_timestamp"])
    )
    num_unique = visits.shape[0]

    # (2) Veraset cuts visits at UTC midnight (4 or 5 pm local). A visit starting in the first MIDNIGHT_WINDOW of a UTC day at the same place as the
    #     device's previous visit, which started the UTC day before and ended at most MAX_SPLIT_GAP earlier, is the rest of that visit, so the two become
    #     one (chains of them too, for multi-day stays). It keeps the first half's start and everything else, and its dwell runs to the end of the last half
    visits = (visits
        .with_columns((pl.col("utc_timestamp") + pl.col("minimum_dwell").cast(pl.Int64) * 60).alias("utc_end"))
        .with_columns(
            (pl.col("utc_timestamp") % 86400 < MIDNIGHT_WINDOW).alias("after_midnight"),
            ((pl.col("safegraph_place_id") == pl.col("safegraph_place_id").shift(1).over([period, "caid_hash"]))
             & (pl.col("utc_timestamp") - pl.col("utc_end").shift(1).over([period, "caid_hash"]) <= MAX_SPLIT_GAP)).fill_null(False).alias("same_place_soon_after"),
            (pl.col("utc_timestamp") // 86400 > pl.col("utc_timestamp").shift(1).over([period, "caid_hash"]) // 86400).fill_null(False).alias("new_utc_day"),
        )
        .with_columns((pl.col("after_midnight") & pl.col("same_place_soon_after") & pl.col("new_utc_day")).alias("continues"))
    )
    # How often a start right after UTC midnight picks up where the previous visit left off, against how often any other start comes that soon after a
    # visit to the same place (the gap between the two is what the merge relies on)
    print(f"starts in the first {MIDNIGHT_WINDOW // 60} min after UTC midnight that continue the previous visit: {visits.filter('after_midnight')['continues'].mean():.3f}",
          f"| other starts within {MAX_SPLIT_GAP // 60} min of a visit to the same place: {visits.filter(~pl.col('after_midnight'))['same_place_soon_after'].mean():.3f}")
    visits = (visits
        .drop(["after_midnight", "same_place_soon_after", "new_utc_day"])
        .with_columns((~pl.col("continues")).cast(pl.UInt32).cum_sum().over([period, "caid_hash"]).alias("visit_number"))
        .group_by([period, "caid_hash", "visit_number"], maintain_order = True)
        .agg(pl.all().exclude(["utc_end", "continues"]).first(), pl.col("utc_end").max(), pl.len().alias("visit_parts"))
        .with_columns(
            ((pl.col("utc_end") - pl.col("utc_timestamp")) / 60).alias("minimum_dwell"),
            (pl.col("utc_timestamp") % 86400 < MIDNIGHT_WINDOW).alias("utc_midnight_start"),
        )
        .drop(["visit_number", "utc_end"])
    )
    num_merged = visits.shape[0]

    # (3) Only visits starting inside the device's window (see device_windows): the local calendar year, or the 60+ day stretch
    visits = (visits
        .join(windows.select([period, "caid_hash", "window_start_day", "window_end_day"]), on = [period, "caid_hash"], how = "inner")
        .filter((pl.col("day") >= pl.col("window_start_day")) & (pl.col("day") <= pl.col("window_end_day")))
        .drop(["window_start_day", "window_end_day"])
    )

    print(f"rows: {num_rows} | duplicates dropped: {num_rows - num_unique} | halves merged back: {num_unique - num_merged} | outside the window: {num_merged - visits.shape[0]} | visits left: {visits.shape[0]}")
    return visits""")

code("""### PART B, SUBPART I
## Write-up: section 0.3 (cleaning counts, yearly sample)
## Output: every visit by a [2ai] device to a dispensary, one row per visit
##         Columns are "year", the visit ("caid_hash", "utc_timestamp", "local_timestamp", "day", "week", "month", "minimum_dwell", the label
##         "safegraph_place_id", "placekey", "location_name", "street_address", "city", "zip_code", "census_block_group"), the WSLCB storefront
##         ("street_address_norm", "city_norm", "tradenames", "operating_licenses") and how it got approved ("address_match", "name_check",
##         "outline_evidence"), and where the store is ("store_cbg", "store_county_fips", "store_county_name", "store_place_fips", "store_place_name",
##         "store_place_type", "store_zcta"), and the cleaning flags "visit_parts", "utc_midnight_start" (see the helper)
## Name: 4bi_dispensary_visits_yearly.parquet
results = []
for YEAR in YEARS:
    visits = dispensary_visits(f"processed-data/sample_restricted_yearly_{YEAR}.parquet")
    results.append(visits.with_columns(pl.lit(int(YEAR)).cast(pl.Int32).alias("year")))

yearly_visits = fix_visits(pl.concat(results), "year", yearly_windows)
yearly_visits = yearly_visits.select(["year"] + [c for c in yearly_visits.columns if c != "year"]).sort(["year", "caid_hash", "local_timestamp"])
yearly_visits.write_parquet("processed-data/4bi_dispensary_visits_yearly.parquet")

(yearly_visits
    .group_by("year")
    .agg(pl.len().alias("visits"), pl.col("caid_hash").n_unique().alias("devices_visiting"), pl.struct(STOREFRONT).n_unique().alias("stores_visited"))
    .join(yearly_crosswalk.group_by("year").len().rename({"len": "devices_with_home"}), on = "year")
    .sort("year"))""")

code("""### PART B, SUBPART II
## Write-up: section 0.3 (cleaning counts, quarterly sample)
## Output: every visit by a [2aii] device to a dispensary during its 60+ day stretch that quarter, one row per visit. Same columns
##         as [4bi], with "quarter" ("2019Q1" etc.) instead of "year"
## Name: 4bii_dispensary_visits_quarterly.parquet
results = []
for YEAR in YEARS:
    for QUARTER in QUARTERS:
        visits = dispensary_visits(f"processed-data/sample_restricted_quarterly_{YEAR}Q{QUARTER}.parquet")
        results.append(visits.with_columns(pl.lit(f"{YEAR}Q{QUARTER}").alias("quarter")))

quarterly_visits = fix_visits(pl.concat(results), "quarter", quarterly_windows)
quarterly_visits = quarterly_visits.select(["quarter"] + [c for c in quarterly_visits.columns if c != "quarter"]).sort(["quarter", "caid_hash", "local_timestamp"])
quarterly_visits.write_parquet("processed-data/4bii_dispensary_visits_quarterly.parquet")

(quarterly_visits
    .group_by("quarter")
    .agg(pl.len().alias("visits"), pl.col("caid_hash").n_unique().alias("devices_visiting"), pl.struct(STOREFRONT).n_unique().alias("stores_visited"))
    .join(quarterly_crosswalk.group_by("quarter").len().rename({"len": "devices_with_home"}), on = "quarter")
    .sort("quarter"))""")

code("""### CHECKS
# (1) Block sums by place should be exactly [3d] (the DHC blocks are made consistent with the places they sit in)
place_check = (blocks_df
    .join(block_places.drop_nulls("place_fips"), on = "block2020", how = "inner")
    .group_by("place_fips").agg(pl.col("population_21plus").sum().alias("blocks_21plus"))
    .join(pl.read_parquet("processed-data/3d_place_population_2020.parquet"), on = "place_fips", how = "full")
)
print("[3d] places:", place_check["place_name"].drop_nulls().shape[0], "| with blocks:", place_check["blocks_21plus"].drop_nulls().shape[0],
      "| total 21+ difference:", (place_check["blocks_21plus"] - place_check["population_21plus"]).abs().sum())

# (2) Every visiting device should be in that period's sample, and nearly all should have a home in [4a]
for name, visits, crosswalk, period in [("[4bi]", yearly_visits, yearly_crosswalk, "year"), ("[4bii]", quarterly_visits, quarterly_crosswalk, "quarter")]:
    device_periods = visits.select(["caid_hash", period]).unique()
    print(name, "device-periods visiting:", device_periods.shape[0], "| without a home in [4a]:", device_periods.join(crosswalk, on = ["caid_hash", period], how = "anti").shape[0])

missing = 0
for YEAR in YEARS:
    sample_devices = pl.read_parquet(f"processed-data/devices_with_350days_{YEAR}.parquet").select("caid_hash")
    missing += yearly_visits.filter(pl.col("year") == int(YEAR)).join(sample_devices, on = "caid_hash", how = "anti").shape[0]
print("[4bi] visits by devices not in [2ai]:", missing)

# (3) No label-month gets more visits from the sample than [1e] has from everybody
over = (yearly_visits.group_by(LABEL_KEY).len()
    .join(store_labels.select(LABEL_KEY), on = LABEL_KEY, how = "inner", nulls_equal = True)
    .join(pl.read_parquet("processed-data/1e_storefront_place_months.parquet").filter(pl.col("approved"))
            .with_columns(pl.col("month").str.replace("-", "").cast(pl.Int32)).unique(LABEL_KEY).select(LABEL_KEY + ["num_visits"]),
          on = LABEL_KEY, how = "inner", nulls_equal = True)
    .filter(pl.col("len") > pl.col("num_visits")))
print("label-months with more sample visits than [1e] total:", over.shape[0])

# (4) The windows: every sample device should have one
for YEAR in YEARS:
    sample_devices = pl.read_parquet(f"processed-data/devices_with_350days_{YEAR}.parquet").select("caid_hash")
    print(YEAR, "[2ai] devices:", sample_devices.shape[0], "| without a window:", sample_devices.join(yearly_windows.filter(pl.col("year") == int(YEAR)), on = "caid_hash", how = "anti").shape[0])

missing = 0
for YEAR in YEARS:
    for QUARTER in QUARTERS:
        sample_devices = pl.read_parquet(f"processed-data/devices_with_60days_{YEAR}Q{QUARTER}.parquet").select("caid_hash")
        missing += sample_devices.join(quarterly_windows.filter(pl.col("quarter") == f"{YEAR}Q{QUARTER}"), on = "caid_hash", how = "anti").shape[0]
print("[2aii] device-quarters without a window:", missing)

# (5) The UTC midnight cut, after cleaning: starts in the 30 minutes after UTC midnight against the average of the half hours on either side. What's left
#     over is visits whose part before midnight was too short to be a visit of its own (their real start is earlier)
for name, visits in [("[4bi]", yearly_visits), ("[4bii]", quarterly_visits)]:
    seconds = visits["utc_timestamp"] % 86400
    after = (seconds < 1800).sum()
    around = (((seconds >= 86400 - 3600) & (seconds < 86400 - 1800)).sum() + ((seconds >= 1800) & (seconds < 3600)).sum()) / 2
    print(name, "starts in the 30 min after UTC midnight:", after, "| half hours either side:", around,
          "| excess, as a share of all visits:", round((after - around) / visits.shape[0], 4), "| visits put back together from 2+ rows:", (visits["visit_parts"] > 1).sum())

# (6) Start times by local hour, daylight time (UTC-7) against standard time (UTC-8), quarterly sample. Anything Veraset does on UTC time would show up an
#     hour apart in the two. The drop at noon and the jump at 7 pm are in the same place in both, so they're local
(quarterly_visits
    .with_columns(
        ((pl.col("local_timestamp") - pl.col("utc_timestamp")) // 3600).cast(pl.String).alias("utc_offset"),
        pl.from_epoch("local_timestamp").dt.hour().alias("hour"),
    )
    .group_by(["utc_offset", "hour"]).len()
    .with_columns((pl.col("len") / pl.col("len").sum().over("utc_offset")).round(3).alias("share"))
    .filter(pl.col("hour").is_between(10, 20))
    .pivot(on = "utc_offset", index = "hour", values = "share")
    .sort("hour"))""")

code("""### PART C, HELPER: SHOWING AND SAVING
CHART_FOLDER = "Charts and Figures"
SAMPLES = [ # (label for titles, label for file names, crosswalk, visits, period column)
    ("Yearly Sample", "yearly", yearly_crosswalk, yearly_visits, "year"),
    ("Quarterly Sample", "quarterly", quarterly_crosswalk, quarterly_visits, "quarter"),
]

def show(fig, name):
    fig.show()
    fig.write_html(f"{CHART_FOLDER}/{name}.html", include_plotlyjs = "cdn") # so they can be opened outside the notebook too""")

code("""### PART C, SUBPART I: GEOGRAPHIC REPRESENTATIVENESS
## Write-up: section 1, figures F1.1-F1.4
# Same as week 6, but with the [4a] home counties (first home visit, see [2b]) and [3c] for the census side. Each device counts once per year/quarter
# it's in the sample. Device shares are out of the devices with a WA home (week 6 divided by every device, WA or not)
wa_counties = (pl.read_parquet("processed-data/3c_county_population_2020.parquet")
    .with_columns((pl.col("population_total") / pl.col("population_total").sum()).alias("population_share"))
)

for sample_title, sample_name, crosswalk, visits, period in SAMPLES:
    county_counts = (crosswalk
        .filter(pl.col("home_county_fips").str.starts_with(STATE_FIPS))
        .group_by(pl.col("home_county_fips").alias("county_fips"))
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum()).alias("device_share"))
        .sort("device_share", descending = True)
    )

    for cut_largest in [False, True]:
        # Optional code to cut out the 5 largest counties
        counts = county_counts.tail(-5) if cut_largest else county_counts

        comparison = (
            counts
            .join(
                wa_counties,
                on = "county_fips",
                how="inner",
            )
        )

        comparison = comparison.sort("population_share", descending = True)

        plot_df = (
            comparison
            .select(
                [
                    "county_name",
                    "device_share",
                    "population_share",
                ]
            )
            .unpivot(
                index="county_name",
                on=["device_share", "population_share"],
                variable_name="source",
                value_name="share",
            )
        )

        plot_df = plot_df.with_columns(
            pl.col("source")
            .replace(
                {
                    "device_share": "Veraset devices sample",
                    "population_share": "2020 Census",
                }
            )
        )

        pdf = plot_df.to_pandas()

        fig3 = px.bar(
            pdf,
            x="county_name",
            y="share",
            color="source",
            barmode="group",
        )

        fig3.update_layout(
            xaxis_title="County",
            yaxis_title="Population Share",
            title = "Geographic Representativeness of Veraset Sample" + (", 5 Largest Counties Excluded" if cut_largest else "") + f", {sample_title}",
            legend_title="",
        )

        fig3.update_xaxes(
            tickangle=45
        )

        show(fig3, f"representativeness_{sample_name}" + ("_5_largest_excluded" if cut_largest else ""))""")

code("""### PART C, SUBPART I: SAMPLE SIZES AND REPRESENTATIVENESS (TABLES)
## Write-up: section 1, tables T1.1-T1.2 and T1.4
## Output: [4ci] one row per sample and period, where the periods are each year/quarter, each year for the quarterly sample (its quarters pooled), and "all"
##         Columns are "sample", "period", "devices" (distinct devices), "device_periods", "with_wa_home" (device-periods with a WA home county in [4a]),
##         "observed_days_mean" (see device_windows), "dissimilarity" (half the sum over WA counties of |device share - 2020 population share|, ie. the share
##         of devices that would have to move county for the sample to match the census), "correlation" (device share vs. population share, across counties),
##         "chi2" (Pearson's chi-squared statistic of the devices' home counties against the census: the sum over counties of (devices - expected)^2 / expected,
##         where expected = WA-home devices x population share), "chi2_p_value", "chi2_per_device" (chi2 / WA-home devices), and both as multiples of the
##         sample's "all" row ("chi2_vs_all", "chi2_per_device_vs_all"). chi2 grows with the number of devices even if the shares don't change, so the
##         per-device version is the one to compare across periods of different sizes
## Name: 4ci_sample_sizes.parquet
## Output: [4ci'] one row per sample, period, and WA county, with "county_fips", "county_name", "devices", "device_share", "population_share", "ratio"
##         (device share / population share)
## Name: 4ci_county_representativeness.parquet
WINDOWS = {"yearly": yearly_windows, "quarterly": quarterly_windows}

def period_groups(df, period): # (label, rows) for each year/quarter, each year of quarters, and everything
    df = df.with_columns(pl.col(period).cast(pl.String).alias("period"))
    groups = [(p, df.filter(pl.col("period") == p)) for p in sorted(df["period"].unique())]
    if period == "quarter":
        groups += [(YEAR, df.filter(pl.col("period").str.starts_with(YEAR))) for YEAR in YEARS]
    return groups + [("all", df)]

size_rows = []
county_rows = []
for sample_title, sample_name, crosswalk, visits, period in SAMPLES:
    devices = WINDOWS[sample_name].join(crosswalk.select([period, "caid_hash", "home_county_fips"]), on = [period, "caid_hash"], how = "left")
    for label, rows in period_groups(devices, period):
        counties = (wa_counties
            .join(rows.group_by(pl.col("home_county_fips").alias("county_fips")).len().rename({"len": "devices"}), on = "county_fips", how = "left")
            .with_columns(pl.col("devices").fill_null(0))
            .with_columns((pl.col("devices") / pl.col("devices").sum()).alias("device_share"))
            .with_columns((pl.col("device_share") / pl.col("population_share")).alias("ratio"))
        )
        county_rows.append(counties
            .select(["county_fips", "county_name", "devices", "device_share", "population_share", "ratio"])
            .with_columns(pl.lit(sample_name).alias("sample"), pl.lit(label).alias("period"))
        )
        num_devices = counties["devices"].sum()
        expected = num_devices * counties["population_share"]
        chi2 = (((counties["devices"] - expected) ** 2) / expected).sum()
        size_rows.append({
            "sample": sample_name, "period": label,
            "devices": rows["caid_hash"].n_unique(),
            "device_periods": rows.shape[0],
            "with_wa_home": rows.filter(pl.col("home_county_fips").str.starts_with(STATE_FIPS)).shape[0],
            "observed_days_mean": rows["observed_days"].mean(),
            "dissimilarity": ((counties["device_share"] - counties["population_share"]).abs().sum() / 2),
            "correlation": counties.select(pl.corr("device_share", "population_share")).item(),
            "chi2": chi2,
            "chi2_p_value": chi2_distribution.sf(chi2, counties.shape[0] - 1),
            "chi2_per_device": chi2 / num_devices,
        })

sample_sizes = (pl.DataFrame(size_rows)
    .with_columns( # the pooled "all" row is the baseline (1.00x)
        (pl.col("chi2") / pl.col("chi2").filter(pl.col("period") == "all").first().over("sample")).alias("chi2_vs_all"),
        (pl.col("chi2_per_device") / pl.col("chi2_per_device").filter(pl.col("period") == "all").first().over("sample")).alias("chi2_per_device_vs_all"),
    )
)
sample_sizes.write_parquet("processed-data/4ci_sample_sizes.parquet")
pl.concat(county_rows).select(["sample", "period", "county_fips", "county_name", "devices", "device_share", "population_share", "ratio"]).write_parquet("processed-data/4ci_county_representativeness.parquet")

pl.Config.set_tbl_rows(40)
sample_sizes.with_columns(pl.col(["observed_days_mean", "dissimilarity", "correlation", "chi2", "chi2_per_device", "chi2_vs_all", "chi2_per_device_vs_all"]).round(3))""")

code("""### PART C, SUBPART I: REPRESENTATIVENESS OVER TIME (CHART)
## Write-up: section 1, figures F1.5-F1.6
# Each year's (quarter's) chi-squared statistic against the census, as a multiple of the one for all years (quarters) pooled. Two lines: the raw statistic,
# which also moves with the number of devices, and the statistic per device, which only moves with how far the county shares are from the census
for sample_title, sample_name, crosswalk, visits, period in SAMPLES:
    over_time = (sample_sizes
        .filter((pl.col("sample") == sample_name) & (pl.col("period").str.len_chars() == (4 if period == "year" else 6))) # years / quarters only
        .sort("period")
        .unpivot(index="period", on=["chi2_vs_all", "chi2_per_device_vs_all"], variable_name="measure", value_name="multiple")
        .with_columns(pl.col("measure").replace({
            "chi2_vs_all": "Chi-squared statistic",
            "chi2_per_device_vs_all": "Chi-squared statistic per device (size-adjusted)",
        }))
    )

    fig13 = px.line(
        over_time.to_pandas(),
        x="period",
        y="multiple",
        color="measure",
        markers=True,
        labels={
            "period": "Year" if period == "year" else "Quarter",
            "multiple": f"Multiple of all {period}s pooled",
            "measure": "",
        },
        title=f"Chi-Squared Distance of Device Home Counties from the 2020 Census<br><sup>Relative to all {period}s pooled (= 1.00x), {sample_title}</sup>",
    )

    fig13.add_hline(y=1, line_dash="dot", annotation_text=f"all {period}s pooled = 1.00x")
    fig13.update_layout(legend=dict(orientation="h", y=-0.25))
    if period == "year":
        fig13.update_xaxes(type="category")

    show(fig13, f"representativeness_chi2_{sample_name}")""")

code("""### PART C, SUBPART II: DISPENSARY VISIT DISTRIBUTIONS
## Write-up: section 2, figures F2.1-F2.4
# Same as week 6 (only devices with at least one visit), counted per device per year/quarter instead of over all three years
for sample_title, sample_name, crosswalk, visits, period in SAMPLES:
    # (5) Graphing overall device dispensary visit distribution

    # Count rows for each device
    group_sizes = (
        visits
        .group_by(["caid_hash", period])
        .len()
        .rename({"len": "group_size"})
    )

    # Count the frequency of each group size and compute proportions
    size_distribution = (
        group_sizes
        .group_by("group_size")
        .len()
        .rename({"len": "count"})
        .sort("group_size")
        .with_columns(
            (pl.col("count") / pl.col("count").sum()).alias("proportion")
        )
    )

    # Convert to pandas for graphing
    df_plot = size_distribution.to_pandas()

    for min_visits in [1, 10, 50]:
        fig4 = px.bar(
            df_plot[df_plot["group_size"] >= min_visits],
            x="group_size",
            y="proportion",
            log_x=True,  # optional
            labels={
                "group_size": f"Number of Visits (per {period})",
                "proportion": "Proportion",
            },
            title="Number of Dispensary Visits by Device" + (f", Restricted to {min_visits}+ visits" if min_visits > 1 else "") + f", {sample_title}",
        )

        show(fig4, f"visits_by_device_{sample_name}" + (f"_{min_visits}plus" if min_visits > 1 else ""))

    # (6) Graphing weekly device dispensary visit distribution

    # Weekly count for each device's dispensary visits
    group_sizes = (
        visits
        .with_columns(
            ((pl.col("local_timestamp") + 4 * 86400) // (7 * 86400)).alias("week") # weeks elapsed, UNIX standard time (week starts on Monday to make weekend calculations easier)
        )
        .group_by(["caid_hash", period, "week"])
        .len()
        .rename({"len": "group_size"})
    )

    # Count the frequency of each group size and compute proportions
    size_distribution = (
        group_sizes
        .group_by("group_size")
        .len()
        .rename({"len": "count"})
        .sort("group_size")
        .with_columns(
            (pl.col("count") / pl.col("count").sum()).alias("proportion")
        )
    )

    # Convert to pandas for graphing
    df_plot = size_distribution.to_pandas()

    for min_visits in [1, 3, 7]:
        fig5 = px.bar(
            df_plot[df_plot["group_size"] >= min_visits],
            x="group_size",
            y="proportion",
            log_x=True,  # optional
            labels={
                "group_size": "Number of Visits",
                "proportion": "Proportion",
            },
            title="Number of Weekly Dispensary Visits by Device" + (f", Restricted to {min_visits}+ visits" if min_visits > 1 else "") + f", {sample_title}",
        )

        show(fig5, f"weekly_visits_by_device_{sample_name}" + (f"_{min_visits}plus" if min_visits > 1 else ""))

    print(sample_title, "| devices with a visit:", visits.select(["caid_hash", period]).unique().shape[0], "of", crosswalk.shape[0], "device-periods with a home")""")

code("""### PART C, SUBPART II: DISPENSARY VISITS (TABLES)
## Write-up: section 2, tables T2.1-T2.3
## Output: [4cii] one row per sample, period (as in [4ci]), and measure, with how many device-periods visit a dispensary and, among those that do, the
##         distribution of the measure
##         Measures are "visits", "visits per 30 observed days" (visits / observed_days * 30), "stores (licenses)" (distinct stores, counting a store by its
##         lowest operating license that month, like Step 5), and "storefronts" (distinct WSLCB addresses)
##         Columns are "sample", "period", "measure", "device_periods" (every sample device, visits or not), "with_visit", "share_with_visit", then over the
##         device-periods with a visit: "mean", "sd", "min", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max"
## Name: 4cii_visit_distributions.parquet
MEASURES = ["visits", "visits per 30 observed days", "stores (licenses)", "storefronts"]

results = []
for sample_title, sample_name, crosswalk, visits, period in SAMPLES:
    devices = (WINDOWS[sample_name]
        .join(visits
                .group_by([period, "caid_hash"])
                .agg(
                    pl.len().alias("visits"),
                    pl.col("operating_licenses").list.min().n_unique().alias("stores (licenses)"),
                    pl.struct(STOREFRONT).n_unique().alias("storefronts"),
                ),
              on = [period, "caid_hash"], how = "left")
        .with_columns(pl.col(["visits", "stores (licenses)", "storefronts"]).fill_null(0))
        .with_columns((pl.col("visits") / pl.col("observed_days") * 30).alias("visits per 30 observed days"))
    )
    for label, rows in period_groups(devices, period):
        visitors = rows.filter(pl.col("visits") > 0)
        for measure in MEASURES:
            results.append(visitors.select(
                pl.lit(sample_name).alias("sample"),
                pl.lit(label).alias("period"),
                pl.lit(measure).alias("measure"),
                pl.lit(rows.shape[0]).alias("device_periods"),
                pl.len().alias("with_visit"),
                (pl.len() / rows.shape[0]).alias("share_with_visit"),
                pl.col(measure).cast(pl.Float64).mean().alias("mean"),
                pl.col(measure).cast(pl.Float64).std().alias("sd"),
                pl.col(measure).cast(pl.Float64).min().alias("min"),
                *[pl.col(measure).cast(pl.Float64).quantile(q).alias(name) for q, name in [(0.10, "p10"), (0.25, "p25"), (0.5, "median"), (0.75, "p75"),
                                                                                           (0.90, "p90"), (0.95, "p95"), (0.99, "p99")]],
                pl.col(measure).cast(pl.Float64).max().alias("max"),
            ))

visit_distributions = pl.concat(results, how = "vertical_relaxed")
visit_distributions.write_parquet("processed-data/4cii_visit_distributions.parquet")

visit_distributions.filter(pl.col("period").is_in(["all", "2019", "2020", "2021"])).with_columns(pl.col(["share_with_visit", "mean", "sd"]).round(3))""")

code("""### PART C, SUBPART III: VISIT START TIMES AND DURATIONS
## Write-up: section 3, figures F3.4-F3.6
SECONDS_PER_DAY = 86400
BIN = 600  # 10 minutes
N_BINS = SECONDS_PER_DAY // BIN

clock_times = [
    f"{(i*10)//60:02d}:{(i*10)%60:02d}"
    for i in range(N_BINS)
]

def occupancy_chart(visits, title, name):
    # Dispensary occupancy rates by time of day
    n_days = (
        visits
        .select(
            (pl.col("local_timestamp") // SECONDS_PER_DAY)
            .n_unique()
        )
        .item()
    )

    # Arrival/departure bins
    bins = (
        visits
        .select(
            arrival_bin=(
                (pl.col("local_timestamp") % SECONDS_PER_DAY) // BIN
            ).cast(pl.UInt16),

            departure_bin=(
                ((pl.col("local_timestamp") + pl.col("minimum_dwell") * 60)
                 % SECONDS_PER_DAY) // BIN
            ).cast(pl.UInt16),

            wraps_midnight=(
                ((pl.col("local_timestamp") + pl.col("minimum_dwell") * 60)
                 % SECONDS_PER_DAY)
                <
                (pl.col("local_timestamp") % SECONDS_PER_DAY)
            )
        )
    )

    diff = np.zeros(N_BINS + 1, dtype=np.int64)

    np.add.at(diff, bins["arrival_bin"].to_numpy(), 1)
    np.add.at(diff, bins["departure_bin"].to_numpy(), -1)
    diff[0] += bins["wraps_midnight"].sum()

    occupancy = np.cumsum(diff[:-1]) / n_days

    fig7 = px.bar(
        x=clock_times,
        y=occupancy,
        labels={
            "x": "Time of day",
            "y": "Average devices present"
        },
        title=title
    )

    fig7.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=clock_times[::6],
            ticktext=clock_times[::6],
        ),
        bargap=0,
    )

    show(fig7, name)

ARRIVAL_BLOCKS = [ # (start hour, end hour, title)
    (0, 4, "12 am to 4 am"),
    (4, 8, "4 am to 8 am"),
    (8, 12, "8 am to 12 pm"),
    (12, 16, "12 pm to 4 pm"),
    (16, 20, "4 pm to 8 pm"),
    (20, 24, "8 pm to 12 am"),
]

for sample_title, sample_name, crosswalk, visits, period in SAMPLES:
    occupancy_chart(visits, f"Average Dispensary Occupancy by Time of Day, {sample_title}", f"occupancy_by_time_{sample_name}")

    # Number of distinct days represented
    n_days = (
        visits
        .select((pl.col("local_timestamp") // SECONDS_PER_DAY).n_unique())
        .item()
    )

    arrival_rates = (
        visits
        .with_columns(
            arrival_bin=(
                (pl.col("local_timestamp") % SECONDS_PER_DAY) // BIN
            )
        )
        .group_by("arrival_bin")
        .len()
        .with_columns(
            arrivals_per_day=pl.col("len") / n_days
        )
        .sort("arrival_bin")
    )

    arrival_bins = pl.DataFrame({"arrival_bin": range(SECONDS_PER_DAY // BIN)})

    arrival_rates = (
        arrival_bins
        .join(arrival_rates, on="arrival_bin", how="left")
        .fill_null(0)
        .with_columns(
            clock_time=pl.col("arrival_bin").map_elements(
                lambda b: f"{(b*10)//60:02d}:{(b*10)%60:02d}",
                return_dtype=pl.String,
            )
        )
    )

    fig8 = px.bar(
        arrival_rates,
        x="clock_time",
        y="arrivals_per_day",
        labels={
            "clock_time": "Time of day",
            "arrivals_per_day": "Average arrivals per day",
        },
        title=f"Average Dispensary Entrances by Time of Day, {sample_title}",
    )

    fig8.update_layout(
        bargap=0,
        xaxis=dict(
            tickmode="array",
            tickvals=arrival_rates["clock_time"][::6],
            ticktext=arrival_rates["clock_time"][::6],
        ),
    )

    show(fig8, f"entrances_by_time_{sample_name}")

    # Visit durations by when the visit started
    for start_hour, end_hour, block_title in ARRIVAL_BLOCKS:
        block_arrivals = (visits
            .filter(((pl.col("local_timestamp") % 86400) < end_hour * 3600) # arrive before the end of the block
                      & ((pl.col("local_timestamp") % 86400) >= start_hour * 3600)) # arrive at or after the start of it
        )

        fig9 = px.histogram(
            x=block_arrivals["minimum_dwell"],
            log_y=True,
            title=f"Dispensary Visit Duration, {block_title}, {sample_title}"
        )

        show(fig9, f"duration_{start_hour:02d}_to_{end_hour:02d}_{sample_name}")

    # DWELL-FILTERED DURATION GRAPH
    MAX_DWELL = 180
    occupancy_chart(visits.filter(pl.col("minimum_dwell") <= MAX_DWELL), f"Average Dispensary Occupancy by Time of Day, Visit duration < 3 hours, {sample_title}",
                    f"occupancy_by_time_under_3_hours_{sample_name}")""")

code("""### PART C, SUBPART III: WHEN PEOPLE SHOP (TABLES AND CHARTS)
## Write-up: section 3, tables T3.1-T3.3, figures F3.1-F3.3
## Output: [4ciii] visit start times (local) by day of week and hour, each sample's periods pooled
##         Columns are "sample", "day_of_week" (1 = Monday .. 7 = Sunday), "day_name", "hour" (0-23), "visits", "share" (of the sample's visits)
## Name: 4ciii_visit_start_times.parquet
## Output: [4ciii'] one row per sample and period (as in [4ci]), with the share of visits starting outside 8 am-midnight and the distribution of minimum dwell
##         times (minutes)
##         Columns are "sample", "period", "visits", "share_outside_8am_midnight" (start hour 0-7), "share_outside_8am_midnight_under_3h" (the same, for
##         visits under 3 hours, which leaves out most employees), "share_utc_midnight_start" (starts that may have been cut, see Part B), then the minimum
##         dwell's "mean", "sd", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max", "share_3h_plus", "share_23h_plus"
## Name: 4ciii_start_times_and_dwell.parquet
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

start_rows = []
dwell_rows = []
for sample_title, sample_name, crosswalk, visits, period in SAMPLES:
    starts = visits.with_columns( # local_timestamp is local time written as if it were UTC, so plain UTC date parts are local
        pl.from_epoch("local_timestamp").dt.weekday().alias("day_of_week"),
        pl.from_epoch("local_timestamp").dt.hour().alias("hour"),
    )

    start_times = (pl.DataFrame({"day_of_week": range(1, 8)}, schema = {"day_of_week": pl.Int8})
        .join(pl.DataFrame({"hour": range(24)}, schema = {"hour": pl.Int8}), how = "cross")
        .join(starts.group_by(["day_of_week", "hour"]).len().rename({"len": "visits"}), on = ["day_of_week", "hour"], how = "left")
        .with_columns(pl.col("visits").fill_null(0))
        .with_columns(
            (pl.col("visits") / pl.col("visits").sum()).alias("share"),
            pl.col("day_of_week").replace_strict(dict(zip(range(1, 8), DAY_NAMES))).alias("day_name"),
            pl.lit(sample_name).alias("sample"),
        )
    )
    start_rows.append(start_times)

    for label, rows in period_groups(starts, period):
        dwell_rows.append(rows.select(
            pl.lit(sample_name).alias("sample"),
            pl.lit(label).alias("period"),
            pl.len().alias("visits"),
            (pl.col("hour") < 8).mean().alias("share_outside_8am_midnight"),
            (pl.col("hour") < 8).filter(pl.col("minimum_dwell") < 180).mean().alias("share_outside_8am_midnight_under_3h"),
            pl.col("utc_midnight_start").mean().alias("share_utc_midnight_start"),
            pl.col("minimum_dwell").mean().alias("mean"),
            pl.col("minimum_dwell").std().alias("sd"),
            *[pl.col("minimum_dwell").quantile(q).alias(name) for q, name in [(0.10, "p10"), (0.25, "p25"), (0.5, "median"), (0.75, "p75"),
                                                                             (0.90, "p90"), (0.95, "p95"), (0.99, "p99")]],
            pl.col("minimum_dwell").max().alias("max"),
            (pl.col("minimum_dwell") >= 180).mean().alias("share_3h_plus"),
            (pl.col("minimum_dwell") >= 23 * 60).mean().alias("share_23h_plus"),
        ))

    # Heatmap of start times, as shares of the sample's visits
    grid = start_times.sort(["day_of_week", "hour"])["share"].to_numpy().reshape(7, 24)
    fig10 = px.imshow(
        grid,
        x=[f"{h:02d}:00" for h in range(24)],
        y=DAY_NAMES,
        labels={
            "x": "Hour of day (local)",
            "y": "Day of week",
            "color": "Share of visits",
        },
        aspect="auto",
        title=f"Dispensary Visit Start Times by Day of Week and Hour, {sample_title}",
    )

    show(fig10, f"start_times_heatmap_{sample_name}")

    # The two margins
    for part, labels in [("hour", "Hour of day (local)"), ("day_name", "Day of week")]:
        margin = (start_times
            .group_by(["hour"] if part == "hour" else ["day_of_week", "day_name"])
            .agg(pl.col("share").sum())
            .sort("hour" if part == "hour" else "day_of_week")
        )

        fig11 = px.bar(
            margin.to_pandas(),
            x=part,
            y="share",
            labels={
                part: labels,
                "share": "Share of visits",
            },
            title=f"Dispensary Visit Start Times by {'Hour of Day' if part == 'hour' else 'Day of Week'}, {sample_title}",
        )

        show(fig11, f"start_times_by_{'hour' if part == 'hour' else 'day'}_{sample_name}")

    # Every visit's duration
    fig12 = px.histogram(
        x=visits["minimum_dwell"],
        log_y=True,
        labels={"x": "Minimum dwell (minutes)"},
        title=f"Dispensary Visit Duration, All Visits, {sample_title}"
    )

    show(fig12, f"duration_all_{sample_name}")

visit_start_times = pl.concat(start_rows).select(["sample", "day_of_week", "day_name", "hour", "visits", "share"])
visit_start_times.write_parquet("processed-data/4ciii_visit_start_times.parquet")

start_times_and_dwell = pl.concat(dwell_rows)
start_times_and_dwell.write_parquet("processed-data/4ciii_start_times_and_dwell.parquet")

start_times_and_dwell.with_columns(pl.col(["share_outside_8am_midnight", "share_outside_8am_midnight_under_3h", "share_utc_midnight_start", "mean", "sd", "share_3h_plus", "share_23h_plus"]).round(4))""")

code("""# (2) Start times as a day of week x hour grid, shares of each sample's visits (in %)
(visit_start_times
    .with_columns((pl.col("share") * 100).round(2))
    .pivot(on = "hour", index = ["sample", "day_name"], values = "share"))""")

md("""### NOTES
- 2010 CBGs vs. places: of WA's 4,783 2010 CBGs, 3,288 sit entirely inside one place (or entirely outside all of them). 805 are mostly outside any place, so their devices get no home city. In 43 the biggest place holds less than half the CBG's 21+ population. About 83% of sample devices have a home city, and about 1.1% have a home city with a share under one half (`home_place_share`).
- Block sums by place match [3d] exactly, so the city split adds no noise on top of the census numbers.
- [2b] found a home for every [2ai] device. 99.6% of home zips are WA ZCTAs in [3a]; the rest are mostly PO-box-only zips.
- Cleaning (Part B): each Veraset month folder runs through the 1st of the next month (UTC), so every visit on the 1st shows up twice, and a few more are plain duplicates in one folder. That's 5,243 rows in the yearly sample and 11,884 in the quarterly one (4.2% and 3.1%). Veraset also cuts visits at UTC midnight (4 or 5 pm local) and picks the rest up as a new visit within ~30 minutes: 2,036 / 6,020 halves got merged back. 319 / 3,681 visits started outside the local calendar year/quarter (the ~8 hours before and ~16 hours after that the UTC folders add), and another 22,180 quarterly visits fell outside the device's 60+ day stretch.
- Visits: [4bi] has 116,273 visits (2019: 25,836, 2020: 42,178, 2021: 48,259) and [4bii] has 335,894. 20,608 of the 87,962 device-years (23.4%) visit a dispensary at least once, against 104,506 of 1,052,148 device-quarters (9.9%).
- The quarterly sample only counts visits in the device's 60+ day stretch (its longest run of consecutive days, the same run Step 2 found), which averages 80.2 days against 86.8 observed days in the whole quarter. Cut to the local calendar, 5,270 device-years have fewer than 350 observed days and 17,482 device-quarters have a 59-day stretch (Step 2 counted the extra UTC hours as days). They stay in the samples.
- Even after merging, starts are ~1.2% (of all visits) too common in the 30 minutes after UTC midnight: visits whose pre-midnight part was too short to be a visit of its own. Their true start is earlier, so the 16:00/17:00 bars in the start time charts run a little high. "utc_midnight_start" flags them.
- Merged visits can run for days (up to ~7 days in the quarterly sample); 52 yearly and 180 quarterly visits last 23+ hours. They're probably employees, people living upstairs, or phones left behind. They also account for a lot of the visits starting before 8 am: 6.5% (yearly) / 7.2% (quarterly) of visits start between midnight and 8 am, against 3.9% / 5.0% of visits under 3 hours.
- Start times have steps at noon (down) and 7 pm (up) local time. They're at the same local clock time in daylight and standard time, so they aren't a UTC artifact.
- 69 [1e] label-months sit at two storefronts that are one store under two WSLCB spellings (see `step1_future_fixes.md`). Each keeps one storefront here, so no visit is counted twice.
- 21 device-quarters in [4bii] have visits but no home (dropped by [2biii]).
- 2021Q4 has 182,486 devices, against 65-96k in every other quarter (from [2aii], still unexplained), and 67,114 visits. Pooled quarterly charts lean on it.
- Chi-squared against the census, as a multiple of all periods pooled: the raw statistic is 0.28-0.41x for single years and 0.07-0.20x for single quarters, which is mostly just their smaller size (the statistic grows with the number of devices). Per device, years run 0.90x (2020) to 1.38x (2021) and quarters 0.94x to 1.63x (2021Q3), drifting away from the census through 2021. Every period differs from the census at p < 0.001, as expected with this many devices.
- Representativeness: device shares are out of devices with a WA home (week 6 divided by every device), and population shares come from [3c] instead of the ChatGPT county table. The dissimilarity index (share of devices that would have to move county to match the census) is 8.7% for the yearly sample and 7.1% for the quarterly one; King County is under-represented (24-25% of devices vs. 29.5% of people) and Pierce, Snohomish, and Thurston over-represented.
""")

code("")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, sys.argv[1])
