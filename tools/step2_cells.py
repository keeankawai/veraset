STEP2_CELLS = [
'''### PART B, INTERMEDIATE WORK: MONTHLY HOME VISIT HISTORY FOR EVERY RELEVANT DEVICE
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

print(home_history.shape, "| devices:", home_history["caid_hash"].n_unique())''',

'''### PART B, SUBPARTS I-III: HELPER
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
    return homes, sample_devices.shape[0], num_dropped''',

'''### PART B, SUBPARTS I-III: YEARLY
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

pl.DataFrame(summary_rows)''',

'''### PART B, SUBPARTS I-III: QUARTERLY
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

pl.DataFrame(summary_rows)''',

'''### CHECKS
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
)''',
]
