"""Writes veraset_devices_processing.ipynb (Step 2). Everything up to the monthly home visit history is the original hand-written notebook, copied here word for word (trailing spaces
included) so every part has a line number to point to; the last five (from the monthly home visit history on) are the ones tools/step2_cells.py appended. Lines starting with "# Write-up:" sit
outside the cells, so they don't change the notebook.
Running it overwrites the notebook (and its outputs) with these cells, and rerunning Step 2 scans all the raw Veraset data, which is slow.
"""
import nbformat as nbf
import sys

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""This document document aims to accomplish the following: 

(a) 

(i) For each year, create a list of devices with at least 350 distinct days of records (label as ([2ai])). 

(ii) For each quarter, create a list of devices with at least 60 consecutive days of records; record start and end dates for each of these devices (label as ([2aii])) 

(b) For all devices in [2ai] and [2aii] 

(i) First, match all devices with a Veraset-encoded home visit to that location. This location will persist for the remainder of that year/quarter. Label as ([2bi]) 

(ii) For devices with no matched Veraset-encoded home visit during the year/quarter, match their home location to their most recent Veraset-encoded home location. Label as ([2bii]) 

(iii) Devices with no previous Veraset-encoded home location will be ignored. 
""")

code("""### IMPORTS AND PRELIMINARIES 
## MEMORY STUFF 
# Stops polars from holding onto freed memory, and forces it to be returned to the OS 
import os
os.environ['_RJEM_MALLOC_CONF'] = "background_thread:true,dirty_decay_ms:0,muzzy_decay_ms:0"
# Useful note to self: polars' .is_in() doesn't run in the streaming engine! 

# Stops Jupyter from storing deallocated variables indefinitely in memory 
%config HistoryManager.max_output_history = 0

## WORKHORSE LIBRARY 
# Enables data streaming of large files 
import polars as pl

## EXTRAS 
# Cleans up some cell outputs 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)""")

# Write-up: section S.a (the yearly sample, 350+ days)
code("""### PART A, SUBPART I 
## Output: a table of devices by year, where each row is a device that has at least 350 distinct days of observations on that year. 
##         Columns are "caid", "caid_hash" (seeded with 42), and "num_days", the number of unique days it appeared 
## Name: devices_with_350days_{YEAR}.parquet 

YEARS = {"2019", "2020", "2021"}

for YEAR in YEARS: 
    results = [] 
    for hash_match in range(10): 
        lf = (pl.scan_parquet(
                f"raw-data/veraset-visits-wa/*{YEAR}*/*.parquet", 
                hive_partitioning = True
            ) 
            .with_columns(
                pl.col("caid").hash(seed=42).alias("caid_hash"), # hashing really reduces memory pressure, although P(any collision) is about 5*10^-6
                (pl.col("local_timestamp") // 86400).alias("day"), 
            )
            .filter(
                (pl.col("caid_hash") % 10) == hash_match
            )
            .select(["caid_hash", "caid", "day"]) 
        )

        num_present = (
            lf.group_by(["caid_hash", "caid"]) 
            .agg(
                pl.col("day").n_unique().alias("num_days")
                )
            .filter(pl.col("num_days") >= 350)
            .collect(engine = "streaming")
        )

        results.append(num_present) 

    combined_aggs_df = pl.concat(results) 

    combined_aggs_df.write_parquet(f"processed-data/devices_with_350days_{YEAR}.parquet") """)

# Write-up: section S.a (the quarterly sample, 60+ consecutive days)
code("""### PART A, SUBPART II 
## Output: a table of devices by year cross quarter, where each row is a device that has at least 60 consecutive days of observations in that quarter. 
##         Columns are "caid", "caid_hash" (seeded with 42), and "longest_run", the length of the longest consecutive stretch it was around for 
## Name: devices_with_60days_{YEAR}{QUARTER}.parquet 
YEARS = ["2019", "2020", "2021"] 
QUARTERS = [1, 2, 3, 4] # will filter by months using this 

### LEFT: 2019Q1, 2019Q2 
for YEAR in YEARS: 
    for QUARTER in QUARTERS: 
        combined_aggs_df = pl.DataFrame() 
        for hash_match in range(3): 
            base = (pl.scan_parquet(
                    f"raw-data/veraset-visits-wa/**/*{YEAR}*/*.parquet", 
                    hive_partitioning = True
                ) 
                .with_columns(
                    pl.col("caid").hash(seed=42).alias("caid_hash"), # hashing really reduces memory pressure, although P(any collision) is about 5*10^-6
                    (pl.col("local_timestamp") // 86400).alias("day"), 
                    (pl.col("local_timestamp") // 604800).alias("week"), 
                    pl.col("month").str.replace("-", "").cast(pl.Int32), # turns "YYYY-MM" -> "YYYYMM" -> int(YYYYMM) to avoid having to keep the string (could rename folders instead of a string replacement, but this keeps it compatible with the rest of the group's folder names)
                )
                .filter( # filtering by quarter proactively 
                    ((pl.col("month") % 100) > 3*QUARTER - 3) & 
                    ((pl.col("month") % 100) <= 3*QUARTER) 
                )
                .filter(
                    (pl.col("caid_hash") % 3) == hash_match
                )
            )

            eligible_ids = (
                base
                .group_by(["caid_hash"]) 
                .agg(
                    pl.col("day").n_unique().alias("num_days")
                )
                .filter(pl.col("num_days") >= 60) # first pass: restrict to 60 days minimum that quarter 
                .select(["caid_hash"])
            )

            days = (
                base
                .join(eligible_ids, on="caid_hash", how="semi")
                .select(["caid_hash", "day"]) 
                .unique()
                .sort(["caid_hash", "day"]) 
            )

            result = ( # SQL "gaps and islands" method / technique 
                days
                .with_columns(
                    run=(
                        pl.col("day").diff().over("caid_hash").ne(1) # adds a True value whenever a jump of more than 1 happens (new run) 
                        .fill_null(True) # makes this True for the first value as well (initially null because nothing before it) 
                        .cast(pl.UInt32) # turns T/F into 1/0 
                        .cum_sum() # running sum, still in the sorted order 
                        .over("caid_hash") 
                    )
                )
                .group_by(["caid_hash", "run"]) # each device/year gets each of its runs grouped 
                .agg(pl.len().alias("run_length")) # and has its length counted 
                .group_by(["caid_hash"]) 
                .agg(pl.col("run_length").max().alias("longest_run")) # then maximum length is picked 
                .filter(pl.col("longest_run") >= 60)
                .collect(engine = "streaming")
            )

            combined_aggs_df = pl.concat([combined_aggs_df, result])

        combined_aggs_df.write_parquet(f"processed-data/devices_with_60days_{YEAR}Q{str(QUARTER)}.parquet")""")

code("""### INTERMEDIATE WORK: CREATING A CROSSWALK FROM CAIDS TO CAID_HASHES
# Note: super disk-intensive, and involves basically duplicating half the dataset to run the operation. Sorry 
# Gathering all relevant hashes 
lf = (pl.scan_parquet(
    f"processed-data/*devices_with_*.parquet", 
    extra_columns = "ignore" # because some have longest runs, others don't 
    )
    .select("caid_hash")
)

relevant_hashes = lf.collect()["caid_hash"].explode().unique().cast(pl.UInt64)

# --- Stage 1: stream matching rows to disk, no dedupe ---
tmp_path = "processed-data/_tmp_matching_caids.parquet"

(
    pl.scan_parquet("raw-data/veraset-visits-wa/**/*.parquet", hive_partitioning=False)
    .select("caid")
    .with_columns(pl.col("caid").hash(seed=42).alias("caid_hash"))
    .filter(pl.col("caid_hash").is_in(relevant_hashes))
    .sink_parquet(tmp_path, engine="streaming")
)

print(pl.scan_parquet(tmp_path).select(pl.len()).collect())   # how many rows matched

# --- Stage 2: dedupe from the temp file, one hash bucket at a time ---
K = 20   # raise if a bucket is still too heavy
parts = []
for k in range(K):
    part = (
        pl.scan_parquet(tmp_path)
        .filter(pl.col("caid_hash") % K == k)
        .group_by("caid_hash")
        .agg(pl.col("caid").first())
        .collect(engine="streaming")
    )
    parts.append(part)

crosswalk = pl.concat(parts)
crosswalk.write_parquet("processed-data/caid_crosswalk_relevant.parquet")

os.remove(tmp_path) """)

# Write-up: section S.a (the sample-restricted visit tables Steps 4-5 read)
code("""### INTERMEDIATE WORK: CREATING A FILTERED DATASET OF SAMPLE-RESTRICTED VISITS ONLY 
## YEAR-FILTERED 
YEARS = {"2019", "2020", "2021"} 

for YEAR in YEARS: 
    allowed_ids = pl.read_parquet(f"processed-data/devices_with_350days_{YEAR}.parquet")["caid"].explode() 
    results = [] 

    for hash_match in range(10): ## MAKE 10!! 
        lf = (pl.scan_parquet(
                f"raw-data/veraset-visits-wa/*{YEAR}*/*.parquet", 
                hive_partitioning = True
            ) 
            .filter(
                pl.col("caid").is_in(allowed_ids)
            )
            .with_columns(
                pl.col("caid").hash(seed=42).alias("caid_hash"), # hashing to reduce memory pressure, although P(any collision) is about 5*10^-6
            )
            .filter(
                (pl.col("caid_hash") % 10) == hash_match
            )
            .with_columns(
                (pl.col("local_timestamp") // 86400).alias("day"), 
                (pl.col("local_timestamp") // 604800).alias("week"), 
                pl.col("month").str.replace("-", "").cast(pl.Int32), # turns "YYYY-MM" -> "YYYYMM" -> int(YYYYMM) to avoid having to keep the string (could rename folders instead of a string replacement, but this keeps it compatible with the rest of the group's folder names)
            )
            .with_columns(
                (pl.col("month") // 100).alias("year")
            )
        )
        results.append(lf)

    combined_aggs_df = pl.concat(results) 

    combined_aggs_df.sink_parquet(f"processed-data/sample_restricted_yearly_{YEAR}.parquet") 
""")

# Write-up: section S.a (same, quarterly)
code("""## QUARTER-FILTERED 
crosswalk = pl.scan_parquet("processed-data/caid_crosswalk_relevant.parquet")  # columns: caid, caid_hash

crosswalk = pl.read_parquet("processed-data/caid_crosswalk_relevant.parquet")  # columns: caid, caid_hash

YEARS = ["2019", "2020", "2021"]
QUARTERS = [1, 2, 3, 4]

for YEAR in YEARS:
    for QUARTER in QUARTERS:
        # month strings for this quarter, matching the raw "YYYY-MM" format
        months = [f"{YEAR}-{m:02d}" for m in range(3 * QUARTER - 2, 3 * QUARTER + 1)]

        # approved devices for this year-quarter -> (caid, caid_hash) pairs via the crosswalk
        approved_hashes = (
            pl.read_parquet(f"processed-data/devices_with_60days_{YEAR}Q{QUARTER}.parquet")["caid_hash"]
            .unique()
            .cast(pl.UInt64)
            .to_frame("caid_hash")
        )
        allowed = crosswalk.join(approved_hashes, on="caid_hash", how="semi").select(["caid", "caid_hash"])

        (
            pl.scan_parquet(
                f"raw-data/veraset-visits-wa/*{YEAR}*/*.parquet",
                hive_partitioning=True,
            )
            .filter(pl.col("month").is_in(months))          # filter on the raw string, before month is rewritten
            .join(allowed.lazy(), on="caid", how="inner")   # keeps approved devices, adds caid_hash
            .with_columns(
                (pl.col("local_timestamp") // 86400).alias("day"),
                (pl.col("local_timestamp") // 604800).alias("week"),
                pl.col("month").str.replace("-", "").cast(pl.Int32),
            )
            .with_columns((pl.col("month") // 100).alias("year"))
            .sink_parquet(
                f"processed-data/sample_restricted_quarterly_{YEAR}Q{QUARTER}.parquet",
                engine="streaming",
            )
        )""")

code("""combined_aggs_df.write_parquet("processed-data/test_save.parquet")""")

code("""test_save = pl.read_parquet("processed-data/sample_restricted_quarterly_2021Q3.parquet")
#test_save.with_columns(pl.col("caid_hash").hash(seed=42).alias("caid_hash"))
test_save""")

code("""### PART B, INTERMEDIATE WORK: MONTHLY HOME VISIT HISTORY FOR EVERY RELEVANT DEVICE
## Output: a table with one row per device per month that has any home visits, giving the most recent home visit that month
##         Columns are "caid_hash" (seeded with 42), "month" (int YYYYMM, the UTC folder month), "last_home_timestamp" (local), "home_cbg" (Veraset's 2010 CBG),
##         "home_zip_code", "num_home_visits"
## Name: home_visits_monthly_relevant.parquet
# Needed for [2bii], since carrying a home location forward means looking at months before the device made it into a sample. Only the devices in the
# crosswalk (anyone in any [2ai]/[2aii] list) are kept, and only their last home visit per month, which keeps this small
crosswalk = pl.read_parquet("processed-data/caid_crosswalk_relevant.parquet")  # columns: caid, caid_hash

MONTHS = sorted(folder for folder in os.listdir("raw-data/veraset-visits-wa") if folder.startswith("month=")) # skips the stray .DS_Store

results = []
for MONTH in MONTHS:
    month_df = (pl.scan_parquet(f"raw-data/veraset-visits-wa/{MONTH}/*.parquet")
        .filter(pl.col("location_name") == "home")
        .join(crosswalk.lazy(), on = "caid", how = "inner") # keeps relevant devices, adds caid_hash
        .group_by("caid_hash")
        .agg(
            pl.col("local_timestamp").max().alias("last_home_timestamp"),
            pl.col("census_block_group").sort_by("local_timestamp").last().alias("home_cbg"), # explicit sort, the raw data isn't in time order
            pl.col("zip_code").sort_by("local_timestamp").last().alias("home_zip_code"),
            pl.len().alias("num_home_visits"),
        )
        .with_columns(pl.lit(int(MONTH.replace("month=", "").replace("-", ""))).cast(pl.Int32).alias("month"))
        .collect(engine = "streaming")
    )
    results.append(month_df)

home_history = pl.concat(results).select(["caid_hash", "month", "last_home_timestamp", "home_cbg", "home_zip_code", "num_home_visits"])
home_history.write_parquet("processed-data/home_visits_monthly_relevant.parquet")

print(home_history.shape, "| devices:", home_history["caid_hash"].n_unique())""")

# Write-up: section 0.4 (the home rule)
code("""### PART B, SUBPARTS I-III: HELPER
## Takes one sample-restricted visit table plus the start of its window, and returns one row per sample device with its home location
##   [2bi]   the device's first home visit inside the window (by local timestamp). That location is used for the whole window
##   [2bii]  no home visit inside the window -> the most recent home visit before the window starts (from home_visits_monthly_relevant)
##   [2biii] neither -> dropped (but counted, so we know how many)
# Columns are "caid_hash", "home_cbg" (2010 CBG), "home_zip_code", "home_county_fips", "home_visit_timestamp" (local), "home_source" ("in_window" / "carried_forward")
import datetime # for turning window start dates into timestamps

def assign_homes(sample_path, window_start):
    window_start_ts = int(datetime.datetime(window_start.year, window_start.month, window_start.day, tzinfo = datetime.timezone.utc).timestamp()) # local_timestamp is local time written as if it were UTC

    sample = pl.scan_parquet(sample_path)
    sample_devices = sample.select("caid_hash").unique().collect(engine = "streaming")

    # [2bi] first home visit inside the window. The sample tables are cut on UTC month folders, so their first ~8 hours are still the previous
    #       period in local time (eg. Dec 31 evening shows up in the January folder), hence the timestamp filter
    in_window = (sample
        .filter((pl.col("location_name") == "home") & (pl.col("local_timestamp") >= window_start_ts))
        .group_by("caid_hash")
        .agg(
            pl.col("census_block_group").sort_by("local_timestamp").first().alias("home_cbg"),
            pl.col("zip_code").sort_by("local_timestamp").first().alias("home_zip_code"),
            pl.col("local_timestamp").min().alias("home_visit_timestamp"),
        )
        .with_columns(pl.lit("in_window").alias("home_source"))
        .collect(engine = "streaming")
    )

    # [2bii] most recent earlier home visit, for everyone left over
    carried_forward = (home_history
        .join(sample_devices.join(in_window, on = "caid_hash", how = "anti"), on = "caid_hash", how = "semi")
        .filter(pl.col("last_home_timestamp") < window_start_ts)
        .sort("last_home_timestamp")
        .group_by("caid_hash")
        .agg(
            pl.col("home_cbg").last(),
            pl.col("home_zip_code").last(),
            pl.col("last_home_timestamp").last().alias("home_visit_timestamp"),
        )
        .with_columns(pl.lit("carried_forward").alias("home_source"))
    )

    homes = (pl.concat([in_window, carried_forward], how = "vertical")
        .with_columns(pl.col("home_cbg").str.slice(0, 5).alias("home_county_fips"))
        .select(["caid_hash", "home_cbg", "home_zip_code", "home_county_fips", "home_visit_timestamp", "home_source"])
        .sort("caid_hash")
    )

    num_dropped = sample_devices.shape[0] - homes.shape[0] # [2biii]
    return homes, sample_devices.shape[0], num_dropped""")

# Write-up: section 0.4 (homes, yearly sample)
code("""### PART B, SUBPARTS I-III: YEARLY
## Output: one row per [2ai] device (350+ days that year) with its home location
## Name: home_locations_yearly_{YEAR}.parquet
YEARS = ["2019", "2020", "2021"]

summary_rows = []
for YEAR in YEARS:
    homes, num_devices, num_dropped = assign_homes(f"processed-data/sample_restricted_yearly_{YEAR}.parquet", datetime.date(int(YEAR), 1, 1))
    homes.write_parquet(f"processed-data/home_locations_yearly_{YEAR}.parquet")
    summary_rows.append({"period": YEAR, "num_devices": num_devices,
                         "in_window": homes.filter(pl.col("home_source") == "in_window").shape[0],
                         "carried_forward": homes.filter(pl.col("home_source") == "carried_forward").shape[0],
                         "dropped": num_dropped})

pl.DataFrame(summary_rows)""")

# Write-up: section 0.4 (homes, quarterly sample; the dropped counts)
code("""### PART B, SUBPARTS I-III: QUARTERLY
## Output: one row per [2aii] device (60+ consecutive days that quarter) with its home location
## Name: home_locations_quarterly_{YEAR}Q{QUARTER}.parquet
QUARTERS = [1, 2, 3, 4]

summary_rows = []
for YEAR in YEARS:
    for QUARTER in QUARTERS:
        homes, num_devices, num_dropped = assign_homes(f"processed-data/sample_restricted_quarterly_{YEAR}Q{QUARTER}.parquet", datetime.date(int(YEAR), 3 * QUARTER - 2, 1))
        homes.write_parquet(f"processed-data/home_locations_quarterly_{YEAR}Q{QUARTER}.parquet")
        summary_rows.append({"period": f"{YEAR}Q{QUARTER}", "num_devices": num_devices,
                             "in_window": homes.filter(pl.col("home_source") == "in_window").shape[0],
                             "carried_forward": homes.filter(pl.col("home_source") == "carried_forward").shape[0],
                             "dropped": num_dropped})

pl.DataFrame(summary_rows)""")

code("""### CHECKS
# (1) Do all the home CBGs show up in the Step 3 population table for 2010 CBGs? (Veraset uses 2010 boundaries)
cbg2010 = pl.read_parquet("processed-data/3b_cbg2010_population_2020.parquet").select("cbg2010")

for file in sorted(f for f in os.listdir("processed-data") if f.startswith("home_locations_")):
    homes = pl.read_parquet(f"processed-data/{file}")
    unmatched = homes.join(cbg2010, left_on = "home_cbg", right_on = "cbg2010", how = "anti")
    print(f"{file:45} devices: {homes.shape[0]:7} | home CBG not in [3b']: {unmatched.shape[0]:5} (non-WA: {(~unmatched['home_cbg'].str.starts_with('53')).sum()})")

# (2) Spot check: a few in-window devices against their raw home rows in the sample table (first home visit on/after Jan 1 2020, local, should match)
check = pl.read_parquet("processed-data/home_locations_yearly_2020.parquet").filter(pl.col("home_source") == "in_window").sample(3, seed = 42)
(pl.scan_parquet("processed-data/sample_restricted_yearly_2020.parquet")
    .filter((pl.col("location_name") == "home") & pl.col("caid_hash").is_in(check["caid_hash"].implode()) & (pl.col("local_timestamp") >= 1577836800)) # 1577836800 = 2020-01-01 00:00
    .sort("local_timestamp")
    .group_by("caid_hash", maintain_order = True)
    .agg(pl.col("census_block_group").first(), pl.col("local_timestamp").first())
    .collect()
    .join(check.select(["caid_hash", "home_cbg", "home_visit_timestamp"]), on = "caid_hash")
)""")

code("""""")

nb["cells"] = cells
nb["metadata"] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'codemirror_mode': {'name': 'ipython', 'version': 3}, 'file_extension': '.py', 'mimetype': 'text/x-python', 'name': 'python', 'nbconvert_exporter': 'python', 'pygments_lexer': 'ipython3', 'version': '3.13.2'}}
nbf.write(nb, sys.argv[1])
