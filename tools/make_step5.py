import nbformat as nbf
import sys

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""This document aims to accomplish the following, once for the yearly sample ([4ai]/[4bi]) and once for the quarterly sample ([4aii]/[4bii]):

(a) For devices with a known home location, report the share of dispensary visits to stores in the device's home county, city, ZIP code, and CBG. Label as ([5a])

(b) For every county, city, ZIP code, and CBG in each year and quarter: how many stores were open (by the WSLCB samples, [1e']), its 21+ population, stores per 10,000 adults 21+, and which density quartile it's in. Label as ([5b])

(c) Every sample device with its number of dispensary visits and the store access of its home county, city, ZIP code, and CBG

(i) for the yearly sample. Label as ([5ci])

(ii) for the quarterly sample. Label as ([5cii])

(d) Is ease of access correlated with visits? Compare the mean, median, 10th, 25th, 75th, and 90th percentiles of visits by county, city, ZIP code, and CBG, split by (1) whether the area has at least one open store or none, and (2) store density (stores per 10,000 adults 21+), in quartiles. Label as ([5d])

Notes:
- Geographies are the census ones from Step 3: counties ([3c]), places ([3d], cities, towns, and CDPs), ZCTAs ([3a]) for ZIP codes, and 2010 CBGs ([3b']), since that's what Veraset uses. A device's home areas come from [4a], a store's from its WSLCB address ([1d])
- "Open" means a license is operating at the storefront that month by the WSLCB samples, whether or not Veraset has the store. Months that only the undated current list vouches for don't count (the store might have opened after 2021)
- Visits are per device per year (yearly sample) or per quarter (quarterly sample), zeros included
""")

code("""### IMPORTS AND PRELIMINARIES
## WORKHORSE LIBRARY
import polars as pl

## CHARTS (same as week 6)
import plotly.express as px

## EXTRAS
# Cleans up some cell outputs
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)

STOREFRONT = ["street_address_norm", "city_norm"]

# (level, the device's home area in [4a], the store's area in [4b])
LEVELS = [
    ("county", "home_county_fips", "store_county_fips"),
    ("city", "home_place_fips", "store_place_fips"),
    ("zip", "home_zip_code", "store_zcta"), # the device's (USPS) home zip against the ZCTA the store sits in, see the notes
    ("cbg", "home_cbg", "store_cbg"),
]

# (sample, period column, [4a], [4b])
SAMPLES = [
    ("yearly", "year", pl.read_parquet("processed-data/4ai_device_home_geography_yearly.parquet"), pl.read_parquet("processed-data/4bi_dispensary_visits_yearly.parquet")),
    ("quarterly", "quarter", pl.read_parquet("processed-data/4aii_device_home_geography_quarterly.parquet"), pl.read_parquet("processed-data/4bii_dispensary_visits_quarterly.parquet")),
]""")

code("""### PART A
## Write-up: section 4, table T4.1
## Output: for each sample, period (plus "all" for everything pooled), and level, the share of dispensary visits that went to a store in the device's
##         home area. Only devices with a home in [4a] count. At the city level, only devices whose home is in a city (place) count, and a visit to a store
##         outside every city is never "in the home city"
##         Columns are "sample", "period", "level", "devices", "visits", "visits_in_home_area", "share_of_visits" (pooled over visits), and
##         "device_average_share" (each device's own share, averaged over devices, so heavy visitors don't dominate)
## Name: 5a_home_area_visit_shares.parquet
results = []
for SAMPLE, PERIOD, crosswalk, visits in SAMPLES:
    with_homes = visits.join(crosswalk, on = ["caid_hash", PERIOD], how = "inner")
    print(SAMPLE, "| visits:", visits.shape[0], "| by devices with a home:", with_homes.shape[0])

    for LEVEL, HOME, STORE in LEVELS:
        level_visits = (with_homes
            .filter(pl.col(HOME).is_not_null())
            .with_columns((pl.col(HOME) == pl.col(STORE)).fill_null(False).alias("in_home_area"))
            .with_columns(pl.col(PERIOD).cast(pl.String).alias("period"))
        )

        for by_period in [True, False]:
            device_shares = (level_visits
                .with_columns(pl.col("period") if by_period else pl.lit("all").alias("period"))
                .group_by(["caid_hash", "period"] if by_period else ["caid_hash", PERIOD, "period"]) # a device in two periods counts twice in "all"
                .agg(pl.len().alias("visits"), pl.col("in_home_area").sum().alias("visits_in_home_area"))
            )
            results.append(device_shares
                .group_by("period")
                .agg(
                    pl.len().alias("devices"),
                    pl.col("visits").sum(),
                    pl.col("visits_in_home_area").sum(),
                    (pl.col("visits_in_home_area") / pl.col("visits")).mean().alias("device_average_share"),
                )
                .with_columns(
                    pl.lit(SAMPLE).alias("sample"),
                    pl.lit(LEVEL).alias("level"),
                    (pl.col("visits_in_home_area") / pl.col("visits")).alias("share_of_visits"),
                )
            )

home_area_shares = (pl.concat(results)
    .select(["sample", "period", "level", "devices", "visits", "visits_in_home_area", "share_of_visits", "device_average_share"])
    .with_columns(pl.col("level").replace_strict({"county": 0, "city": 1, "zip": 2, "cbg": 3}).alias("level_order"))
    .sort(["sample", "period", "level_order"], descending = [True, False, False])
    .drop("level_order")
)
home_area_shares.write_parquet("processed-data/5a_home_area_visit_shares.parquet")

home_area_shares.filter(pl.col("period") == "all")""")

code("""# (2) The same thing year by year / quarter by quarter, as shares of visits
(home_area_shares
    .filter(pl.col("period") != "all")
    .pivot(on = "level", index = ["sample", "period"], values = "share_of_visits")
    .with_columns(pl.col(["county", "city", "zip", "cbg"]).round(3)))""")

code("""### PART B
## Write-up: section 5.1 (open stores, density, quartiles), table T5.1
## Output: one row per level x area x period (every area in Step 3, stores or not), with how many stores were open there
##         Columns are "sample" ("yearly" / "quarterly"), "period", "level", "area" (county FIPS, place FIPS, ZCTA, or 2010 CBG), "population_21plus",
##         "open_stores" (stores open in an average month of the period), "has_store" (at least one store open at some point in the period),
##         "stores_per_10k" (open_stores per 10,000 adults 21+), "density_quartile" (1 = lowest to 4 = highest, among the areas with a store that period,
##         each area counting once. Null for areas with no store)
## Name: 5b_area_store_access.parquet
# (1) Where each storefront is. A storefront has a row in [1d] for each license there, all with the same coordinates
store_areas = (pl.read_parquet("processed-data/1d_wslcb_location_coordinates.parquet")
    .sort("license")
    .group_by(STOREFRONT, maintain_order = True)
    .agg(pl.col(["county_fips", "place_fips", "zcta", "cbg2010"]).first())
    .rename({"county_fips": "county", "place_fips": "city", "zcta": "zip", "cbg2010": "cbg"})
)

# (2) Stores open each month. A store whose WSLCB address is spelled two ways (or that moved) can show up at two storefronts in the same month, so a
#     store is counted by its license: storefronts sharing their (lowest) operating license that month count once
open_months = (pl.read_parquet("processed-data/1e_storefront_months.parquet")
    .filter(~pl.col("current_list_only"))
    .with_columns(pl.col("operating_licenses").list.min().alias("store_id"))
    .join(store_areas, on = STOREFRONT, how = "left")
    .with_columns(
        pl.col("month").str.slice(0, 4).alias("year"),
        (pl.col("month").str.slice(0, 4) + "Q" + ((pl.col("month").str.slice(5, 2).cast(pl.Int32) - 1) // 3 + 1).cast(pl.String)).alias("quarter"),
    )
)
print("storefront-months open:", open_months.shape[0], "| storefronts:", open_months.select(STOREFRONT).unique().shape[0],
      "| without an area:", open_months.filter(pl.col("county").is_null()).shape[0])
print("stores open per month (min, max):", open_months.group_by("month").agg(pl.col("store_id").n_unique())["store_id"].min(),
      open_months.group_by("month").agg(pl.col("store_id").n_unique())["store_id"].max())

# (3) Every area and its 21+ population, from Step 3
areas = pl.concat([
    pl.read_parquet("processed-data/3c_county_population_2020.parquet").select([pl.lit("county").alias("level"), pl.col("county_fips").alias("area"), "population_21plus"]),
    pl.read_parquet("processed-data/3d_place_population_2020.parquet").select([pl.lit("city").alias("level"), pl.col("place_fips").alias("area"), "population_21plus"]),
    pl.read_parquet("processed-data/3a_zcta_population_2020.parquet").select([pl.lit("zip").alias("level"), pl.col("zcta").alias("area"), "population_21plus"]),
    pl.read_parquet("processed-data/3b_cbg2010_population_2020.parquet").select([pl.lit("cbg").alias("level"), pl.col("cbg2010").alias("area"), "population_21plus"]),
], how = "vertical_relaxed")

# (4) Average open stores per month in each period, for every area (zeros included), then density and quartiles
results = []
for SAMPLE, PERIOD, months_per_period in [("yearly", "year", 12), ("quarterly", "quarter", 3)]:
    periods = open_months.select(PERIOD).unique().rename({PERIOD: "period"})
    for LEVEL, HOME, STORE in LEVELS:
        open_stores = (open_months
            .filter(pl.col(LEVEL).is_not_null()) # stores outside every city don't count toward any city
            .group_by([LEVEL, "month", PERIOD])
            .agg(pl.col("store_id").n_unique().alias("stores"))
            .group_by([LEVEL, PERIOD])
            .agg((pl.col("stores").sum() / months_per_period).alias("open_stores"))
            .rename({LEVEL: "area", PERIOD: "period"})
        )
        results.append(areas
            .filter(pl.col("level") == LEVEL)
            .join(periods, how = "cross")
            .join(open_stores, on = ["area", "period"], how = "left")
            .with_columns(pl.col("open_stores").fill_null(0), pl.lit(SAMPLE).alias("sample"))
        )

area_access = (pl.concat(results)
    .with_columns(
        (pl.col("open_stores") > 0).alias("has_store"),
        (pl.col("open_stores") / (pl.col("population_21plus") / 10000)).alias("stores_per_10k"), # infinite for the odd store in an area nobody lives in
    )
    .with_columns( # quartiles among the areas with a store, ranked so the infinite ones land in the top quartile
        pl.when(pl.col("has_store"))
          .then((4 * pl.col("stores_per_10k").rank("ordinal").over(["sample", "period", "level", "has_store"]) - 1) // pl.len().over(["sample", "period", "level", "has_store"]) + 1)
          .cast(pl.Int32)
          .alias("density_quartile")
    )
    .select(["sample", "period", "level", "area", "population_21plus", "open_stores", "has_store", "stores_per_10k", "density_quartile"])
    .sort(["sample", "period", "level", "area"], descending = [True, False, False, False])
)
area_access.write_parquet("processed-data/5b_area_store_access.parquet")

# Quartile cutoffs (stores per 10k adults 21+), pooled over the periods
(area_access
    .filter(pl.col("has_store"))
    .group_by(["sample", "level", "density_quartile"])
    .agg(pl.len().alias("area_periods"), pl.col("stores_per_10k").min().alias("from"), pl.col("stores_per_10k").filter(pl.col("stores_per_10k").is_finite()).max().alias("to"))
    .with_columns(pl.col("level").replace_strict({"county": 0, "city": 1, "zip": 2, "cbg": 3}).alias("level_order"))
    .sort(["sample", "level_order", "density_quartile"], descending = [True, False, False])
    .drop("level_order"))""")

code("""### PART C, SUBPARTS I-II
## Write-up: section 5.1 (each device's home access)
## Output: one row per sample device per year/quarter (everyone in [4a], zeros included) with its number of dispensary visits and its home areas' access
##         Columns are "year" / "quarter", "caid_hash", "num_visits", the home areas from [4a] ("home_county_fips", "home_place_fips", "home_zip_code",
##         "home_cbg"), and for each level ("county_", "city_", "zip_", "cbg_"): "has_store", "open_stores", "stores_per_10k", "density_quartile".
##         A level's columns are null when the device has no home area at that level (no city, or a home zip that isn't a WA ZCTA)
## Name: 5ci_device_visits_access_yearly.parquet, 5cii_device_visits_access_quarterly.parquet
device_access = {}
for SAMPLE, PERIOD, crosswalk, visits in SAMPLES:
    devices = (crosswalk
        .select([PERIOD, "caid_hash", "home_county_fips", "home_place_fips", "home_zip_code", "home_cbg"])
        .join(visits.group_by(["caid_hash", PERIOD]).len().rename({"len": "num_visits"}), on = ["caid_hash", PERIOD], how = "left")
        .with_columns(pl.col("num_visits").fill_null(0))
    )
    for LEVEL, HOME, STORE in LEVELS:
        level_access = (area_access
            .filter((pl.col("sample") == SAMPLE) & (pl.col("level") == LEVEL))
            .select([
                pl.col("period").alias(PERIOD) if PERIOD == "quarter" else pl.col("period").cast(pl.Int32).alias(PERIOD),
                pl.col("area").alias(HOME),
                *[pl.col(c).alias(f"{LEVEL}_{c}") for c in ["has_store", "open_stores", "stores_per_10k", "density_quartile"]],
            ])
        )
        devices = devices.join(level_access, on = [PERIOD, HOME], how = "left")

    devices.write_parquet(f"processed-data/5c{'i' if SAMPLE == 'yearly' else 'ii'}_device_visits_access_{SAMPLE}.parquet")
    device_access[SAMPLE] = devices
    print(SAMPLE, "| device-periods:", devices.shape[0], "| with a visit:", (devices["num_visits"] > 0).sum(),
          "| without access at the level (county, city, zip, cbg):", [devices[f"{LEVEL}_has_store"].null_count() for LEVEL, _, _ in LEVELS])""")

code("""### PART D
## Write-up: section 5, tables T5.2-T5.3
## Output: the distribution of visits per device-period for each sample, level, and access group, once over every device and once over just the devices
##         with at least one visit
##         Columns are "sample", "devices" ("all" / "1+ visits"), "level", "split" ("open store" / "density"), "group" ("none", "1+ stores", or
##         "Q1 (lowest)".."Q4 (highest)"), "device_periods", "areas" (how many distinct home areas they live in), "share_with_visit", "mean",
##         "p10", "p25", "median", "p75", "p90"
## Name: 5d_visits_by_access.parquet
def visit_stats(df, group):
    return (df
        .group_by(group)
        .agg(
            pl.len().alias("device_periods"),
            (pl.col("num_visits") > 0).mean().alias("share_with_visit"),
            pl.col("num_visits").mean().alias("mean"),
            pl.col("num_visits").quantile(0.10).alias("p10"),
            pl.col("num_visits").quantile(0.25).alias("p25"),
            pl.col("num_visits").median().alias("median"),
            pl.col("num_visits").quantile(0.75).alias("p75"),
            pl.col("num_visits").quantile(0.90).alias("p90"),
        )
        .rename({group: "group"})
    )

results = []
for SAMPLE, PERIOD, crosswalk, visits in SAMPLES:
    for who in ["all", "1+ visits"]:
        devices = device_access[SAMPLE] if who == "all" else device_access[SAMPLE].filter(pl.col("num_visits") > 0)
        for LEVEL, HOME, STORE in LEVELS:
            level_devices = (devices
                .filter(pl.col(f"{LEVEL}_has_store").is_not_null())
                .with_columns(
                    pl.when(pl.col(f"{LEVEL}_has_store")).then(pl.lit("1+ stores")).otherwise(pl.lit("none")).alias("open store"),
                    pl.when(pl.col(f"{LEVEL}_has_store"))
                      .then(pl.lit("Q") + pl.col(f"{LEVEL}_density_quartile").cast(pl.String) + pl.col(f"{LEVEL}_density_quartile").replace_strict({1: " (lowest)", 4: " (highest)"}, default = ""))
                      .otherwise(pl.lit("none"))
                      .alias("density"),
                )
            )
            for split in ["open store", "density"]:
                results.append(visit_stats(level_devices, split)
                    .join(level_devices.group_by(pl.col(split).alias("group")).agg(pl.col(HOME).n_unique().alias("areas")), on = "group")
                    .with_columns(pl.lit(SAMPLE).alias("sample"), pl.lit(who).alias("devices"), pl.lit(LEVEL).alias("level"), pl.lit(split).alias("split"))
                )

visits_by_access = (pl.concat(results)
    .select(["sample", "devices", "level", "split", "group", "device_periods", "areas", "share_with_visit", "mean", "p10", "p25", "median", "p75", "p90"])
    .with_columns(
        pl.col("level").replace_strict({"county": 0, "city": 1, "zip": 2, "cbg": 3}).alias("level_order"),
        pl.col("split").replace_strict({"open store": 0, "density": 1}).alias("split_order"),
        pl.col("group").replace_strict({"none": 0, "1+ stores": 1, "Q1 (lowest)": 1, "Q2": 2, "Q3": 3, "Q4 (highest)": 4}).alias("group_order"),
    )
    .sort(["sample", "devices", "level_order", "split_order", "group_order"], descending = [True, False, False, False, False])
    .drop(["level_order", "split_order", "group_order"])
    .with_columns(pl.col(["share_with_visit", "mean"]).round(3))
)
visits_by_access.write_parquet("processed-data/5d_visits_by_access.parquet")""")

code("""# (2) Yearly sample, every device (visits per device per year)
visits_by_access.filter((pl.col("sample") == "yearly") & (pl.col("devices") == "all")).drop(["sample", "devices"])""")

code("""# (3) Yearly sample, devices with at least one visit that year
visits_by_access.filter((pl.col("sample") == "yearly") & (pl.col("devices") == "1+ visits")).drop(["sample", "devices"])""")

code("""# (4) Quarterly sample, every device (visits per device per quarter)
visits_by_access.filter((pl.col("sample") == "quarterly") & (pl.col("devices") == "all")).drop(["sample", "devices"])""")

code("""# (5) Quarterly sample, devices with at least one visit that quarter
visits_by_access.filter((pl.col("sample") == "quarterly") & (pl.col("devices") == "1+ visits")).drop(["sample", "devices"])""")

code("""### CHECKS
## Write-up: sections 2b, 4a, 5.2c, 6a
# (1) Coverage: the share of open storefront-months (by [1e']) with an approved Veraset label in [1e], by year. Visits can only be counted at those
covered = pl.read_parquet("processed-data/1e_storefront_place_months.parquet").filter(pl.col("approved")).select(STOREFRONT + ["month"]).unique()
print((open_months
    .select(STOREFRONT + ["month", "year"]).unique()
    .join(covered.with_columns(pl.lit(True).alias("covered")), on = STOREFRONT + ["month"], how = "left")
    .group_by("year")
    .agg(pl.len().alias("open_storefront_months"), pl.col("covered").fill_null(False).mean().alias("share_covered"))
    .sort("year")).rows())

# (2) How often the visit's own Veraset zip is the ZCTA the store sits in (the ZIP level compares the home's USPS zip against the store's ZCTA)
for SAMPLE, PERIOD, crosswalk, visits in SAMPLES:
    print(SAMPLE, "| visit zip = store ZCTA:", round((visits["zip_code"] == visits["store_zcta"]).mean(), 3))

# (3) Clark County (across the river from Portland, whose stores never show up in WA-only data) against everyone else, yearly sample
(device_access["yearly"]
    .with_columns(pl.when(pl.col("home_county_fips") == "53011").then(pl.lit("Clark")).otherwise(pl.lit("rest of WA")).alias("home"))
    .group_by("home")
    .agg(pl.len().alias("device_years"), (pl.col("num_visits") > 0).mean().alias("share_with_visit"), pl.col("num_visits").mean().alias("mean")))""")

code("""### CHARTS
## Write-up: figures F4.1, F5.1-F5.4. Saved to "Charts and Figures" as .html files, like Step 4
CHART_FOLDER = "Charts and Figures"
LEVEL_NAMES = {"county": "County", "city": "City", "zip": "ZIP code", "cbg": "CBG"}

def show(fig, name):
    fig.show()
    fig.write_html(f"{CHART_FOLDER}/{name}.html", include_plotlyjs = "cdn")

# (1) Share of visits to stores in the home area
plot_df = (home_area_shares
    .filter(pl.col("period") == "all")
    .with_columns(pl.col("level").replace_strict(LEVEL_NAMES), pl.col("sample").str.to_titlecase())
)

fig1 = px.bar(
    plot_df.to_pandas(),
    x="level",
    y="share_of_visits",
    color="sample",
    barmode="group",
    labels={
        "level": "Home area",
        "share_of_visits": "Share of dispensary visits",
        "sample": "",
    },
    title="Share of Dispensary Visits to a Store in the Device's Home Area",
)

show(fig1, "home_area_visit_shares")

# (2) Share of devices visiting, and mean visits, by the home area's store access
for SAMPLE, PERIOD, crosswalk, visits in SAMPLES:
    plot_df = (visits_by_access
        .filter((pl.col("sample") == SAMPLE) & (pl.col("devices") == "all") & (pl.col("split") == "density"))
        .with_columns(pl.col("level").replace_strict(LEVEL_NAMES))
    )

    for measure, measure_title in [("share_with_visit", "Share of Devices with a Dispensary Visit"), ("mean", "Mean Dispensary Visits per Device")]:
        fig2 = px.bar(
            plot_df.to_pandas(),
            x="group",
            y=measure,
            facet_col="level",
            labels={
                "group": "Stores per 10,000 adults 21+ in the home area",
                "share_with_visit": "Share with a visit",
                "mean": f"Mean visits per device per {PERIOD}",
            },
            title=f"{measure_title} by Store Density in the Home Area (Stores per 10,000 Adults 21+, Quartiles), {SAMPLE.title()} Sample",
        )

        fig2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig2.update_xaxes(title_text="") # one per facet otherwise, all on top of each other (the x-axis is in the title)

        show(fig2, f"{measure}_by_density_{SAMPLE}")""")

md("""### NOTES
- Visits to stores in the home area (pooled, devices with a home): county 87% of visits, city 36% (yearly) / 39% (quarterly) of visits by devices that live in a city, ZIP 32%, CBG 4% (yearly) / 5% (quarterly). The shares hold steady from period to period (2021Q4 is a bit higher for city and CBG).
- ZIP is the device's home zip (Veraset's USPS zip) against the ZCTA the store's WSLCB address sits in. The visit's own Veraset zip matches the store's ZCTA on 90% of visits.
- Stores open per month: 472-503 (licenses operating by the dated WSLCB samples). Areas with an open store in 2020: 38 of 39 counties (not Garfield; Franklin has none from 2021Q2 on), 166 of 639 places, 217 of 606 ZCTAs, and 392 of 4,783 2010 CBGs. About 68 stores in an average month sit outside any city or CDP, so they count toward their county, ZIP, and CBG but no city.
- Most device-periods have no dispensary visit (77% of device-years, 90% of device-quarters), so the 10th-75th percentiles are 0 in nearly every group. The differences show up in the share with a visit and the mean, and in the "1+ visits" tables (how often visitors go).
- Pattern: in the yearly sample, the share of devices visiting rises with density at the city level (no store 19%, Q1 23%, Q4 38%), the ZIP level (19%, 22%, 33%), and less steadily at the CBG level (23% with no store, 29-36% with one). The quarterly sample shows the same thing at lower levels (city: 8%, 10%, 16%). Among devices that do visit, frequency barely moves (median 2 a year in every group).
- Counties don't follow the pattern because Q1 is the big metro counties (King, Pierce, Snohomish, Clark), which have few stores per adult but plenty of stores nearby. Clark County also borders Portland: Oregon dispensaries never show up in WA-only Veraset data, and 18% of Clark devices visit a WA store in a year against 24% elsewhere.
- Coverage: visits only count at stores Veraset has (approved [1e] labels, 62% of open store-months), while access counts every open WSLCB store. An area whose stores aren't in Veraset looks like an area where nobody visits, which pulls the "1+ stores" groups toward zero.
- These are descriptive comparisons only. Urban/rural, income, and everything else that goes with store density aren't controlled for.
""")

code("")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, sys.argv[1])
