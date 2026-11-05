import nbformat as nbf
import sys

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""This document aims to accomplish the following (all population counts come from the 2020 Decennial Census, DHC table P12 "Sex by Age"):

(a) For all ZCTAs (census approximations of zip codes) in WA, list their total population and 21+ population. Label as ([3a])

(b) For all CBGs in WA, list their total population and 21+ population. Label as ([3b])

(b') Same as (b), but on 2010 CBG boundaries, since that's what Veraset's census_block_group column uses. Label as ([3b'])

(c) For all counties in WA, list their total population and 21+ population. Label as ([3c])

(d) For all census places in WA (incorporated cities/towns AND census-designated places), list their total population and 21+ population. Label as ([3d])

Notes:
- The 2020 Census applies differential privacy noise to everything below the state level, so each geography is pulled directly at its own level (rather than summed up from blocks) to keep things as accurate as possible. The only exception is [3b'], which has to be built from blocks.
- The Census API key is read from census_api_key.txt so it isn't sitting in the notebook.
"""))

cells.append(nbf.v4.new_code_cell("""### IMPORTS AND PRELIMINARIES
## WORKHORSE LIBRARY
import polars as pl

## EXTRAS
# For talking to the Census API and grabbing the relationship file
import os
import json
import zipfile
import urllib.request
import urllib.parse

# Cleans up some cell outputs
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

CENSUS_KEY = open("census_api_key.txt").read().strip()
STATE_FIPS = "53" # Washington
os.makedirs("raw-data/census", exist_ok = True)

# All 49 variables of table P12. 001 is the total, 003-008 and 027-032 are the male and female age bins from "Under 5" up to "20 years",
# so 21+ is the total minus those 12 bins (P12 has a standalone "21 years" bin, so no interpolation is needed)
P12_VARS = [f"P12_{i:03d}N" for i in range(1, 50)]
UNDER_21_VARS = [f"P12_{i:03d}N" for i in list(range(3, 9)) + list(range(27, 33))] """))

cells.append(nbf.v4.new_code_cell("""### HELPER: PULLING TABLE P12 FROM THE CENSUS API
## Output: a polars DataFrame with the geography columns the API hands back, plus "population_total" and "population_21plus"
def get_census(geo_for, geo_in = None, include_name = True):
    get_vars = (["NAME"] if include_name else []) + P12_VARS # NAME + 49 variables = 50, which is exactly the API's limit
    params = [("get", ",".join(get_vars)), ("for", geo_for)]
    for geo in (geo_in or []):
        params.append(("in", geo)) # "in" gets repeated for nested geographies (state, then county, then tract...)
    params.append(("key", CENSUS_KEY))

    url = "https://api.census.gov/data/2020/dec/dhc?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as response:
        rows = json.loads(response.read())

    return (pl.DataFrame(rows[1:], schema = rows[0], orient = "row")
        .with_columns(pl.col(P12_VARS).cast(pl.Int64))
        .with_columns(
            pl.col("P12_001N").alias("population_total"),
            (pl.col("P12_001N") - pl.sum_horizontal(UNDER_21_VARS)).alias("population_21plus"),
        )
        .drop(P12_VARS)
    )"""))

cells.append(nbf.v4.new_code_cell("""### PART A
## Output: a table of WA ZCTAs, where each row is a ZCTA with its total and 21+ population
##         Columns are "zcta", "name", "population_total", "population_21plus"
## Name: 3a_zcta_population_2020.parquet
# Note: ZCTAs aren't quite the same thing as USPS zip codes (Veraset's zip_code column), they're built out of census blocks. Most line up,
#       but PO-box-only zips have no ZCTA at all. ZCTAs crossing the state line only get their WA part counted here.
zcta_df = (get_census("zip code tabulation area (or part):*", [f"state:{STATE_FIPS}"])
    .select([
        pl.col("zip code tabulation area (or part)").alias("zcta"),
        pl.col("NAME").alias("name"),
        "population_total",
        "population_21plus",
    ])
    .sort("zcta")
)

zcta_df.write_parquet("processed-data/3a_zcta_population_2020.parquet")

zcta_df"""))

cells.append(nbf.v4.new_code_cell("""### PART B
## Output: a table of WA CBGs (2020 boundaries), where each row is a CBG with its total and 21+ population
##         Columns are "cbg2020" (12-digit GEOID), "county_fips", "population_total", "population_21plus"
## Name: 3b_cbg2020_population_2020.parquet
cbg2020_df = (get_census("block group:*", [f"state:{STATE_FIPS}", "county:*", "tract:*"])
    .select([
        (pl.col("state") + pl.col("county") + pl.col("tract") + pl.col("block group")).alias("cbg2020"),
        (pl.col("state") + pl.col("county")).alias("county_fips"),
        "population_total",
        "population_21plus",
    ])
    .sort("cbg2020")
)

cbg2020_df.write_parquet("processed-data/3b_cbg2020_population_2020.parquet")

cbg2020_df"""))

cells.append(nbf.v4.new_code_cell("""### PART C
## Output: a table of WA counties, where each row is a county with its total and 21+ population
##         Columns are "county_fips" (5 digits), "county_name", "population_total", "population_21plus"
## Name: 3c_county_population_2020.parquet
county_df = (get_census("county:*", [f"state:{STATE_FIPS}"])
    .select([
        (pl.col("state") + pl.col("county")).alias("county_fips"),
        pl.col("NAME").str.replace(" County, Washington", "").alias("county_name"),
        "population_total",
        "population_21plus",
    ])
    .sort("county_fips")
)

county_df.write_parquet("processed-data/3c_county_population_2020.parquet")

county_df"""))

cells.append(nbf.v4.new_code_cell("""### PART D
## Output: a table of WA census places, where each row is a place with its total and 21+ population
##         Columns are "place_fips" (7 digits, state + place), "place_name", "place_type" (city, town, or CDP), "population_total", "population_21plus"
## Name: 3d_place_population_2020.parquet
# Note: places don't cover the whole state (unincorporated areas outside any CDP belong to no place), so these won't sum to the state total
place_df = (get_census("place:*", [f"state:{STATE_FIPS}"])
    .with_columns(
        pl.col("NAME").str.replace(", Washington", "").alias("full_name") # e.g. "Aberdeen city" or "Addy CDP"
    )
    .select([
        (pl.col("state") + pl.col("place")).alias("place_fips"),
        pl.col("full_name").str.replace(r" (city|town|CDP)( \\(.*\\))?$", "$2").alias("place_name"), # a few CDP names get reused, so those keep their "(X County)" tag
        pl.col("full_name").str.extract(r" (city|town|CDP)( \\(.*\\))?$", 1).alias("place_type"),
        "population_total",
        "population_21plus",
    ])
    .sort("place_fips")
)

place_df.write_parquet("processed-data/3d_place_population_2020.parquet")

place_df"""))

cells.append(nbf.v4.new_code_cell("""### INTERMEDIATE WORK: PULLING TABLE P12 AT THE 2020 BLOCK LEVEL
# Needed for [3b'], because 2020 counts can only be moved onto 2010 boundaries from the block level up.
# The API only gives blocks one county at a time, and the big counties take close to a minute, so the result gets cached in raw-data/census
BLOCKS_PATH = "raw-data/census/p12_blocks_2020.parquet"

if not os.path.exists(BLOCKS_PATH):
    results = []
    for county in county_df["county_fips"].str.slice(2, 3):
        block_df = (get_census("block:*", [f"state:{STATE_FIPS}", f"county:{county}", "tract:*"], include_name = False)
            .select([
                (pl.col("state") + pl.col("county") + pl.col("tract") + pl.col("block")).alias("block2020"),
                "population_total",
                "population_21plus",
            ])
        )
        results.append(block_df)

    pl.concat(results).write_parquet(BLOCKS_PATH)

blocks_df = pl.read_parquet(BLOCKS_PATH)
print(blocks_df.shape, blocks_df["population_total"].sum())"""))

cells.append(nbf.v4.new_code_cell("""### PART B'
## Output: a table of WA CBGs (2010 boundaries, to match Veraset), where each row is a CBG with its 2020 total and 21+ population
##         Columns are "cbg2010" (12-digit GEOID), "county_fips", "population_total", "population_21plus" (floats, since blocks get split by area)
## Name: 3b_cbg2010_population_2020.parquet
# Method: the Census 2010-to-2020 block relationship file lists every piece where a 2010 block and a 2020 block overlap, plus the land area of
#         that overlap. Each 2020 block's population gets split across the 2010 blocks it overlaps in proportion to land area, then summed up into 2010 CBGs
#         (the CBG is the first digit of the 4-digit block number). Blocks are small enough that area-weighting barely matters, most 2020 blocks sit in one 2010 block.
REL_PATH = "raw-data/census/TAB2010_TAB2020_ST53.zip"

if not os.path.exists(REL_PATH):
    urllib.request.urlretrieve("https://www2.census.gov/geo/docs/maps-data/data/rel2020/t10t20/TAB2010_TAB2020_ST53.zip", REL_PATH)

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

# Sanity check: the weights for each 2020 block should add up to 1
print("weight sums per 2020 block (min, max):", rel_df.group_by("block2020").agg(pl.col("weight").sum())["weight"].min(),
      rel_df.group_by("block2020").agg(pl.col("weight").sum())["weight"].max())

allocated = (blocks_df
    .join(rel_df.select(["block2020", "cbg2010", "STATE_2010", "weight"]), on = "block2020", how = "left")
    .with_columns(
        (pl.col("population_total") * pl.col("weight")).alias("population_total"),
        (pl.col("population_21plus") * pl.col("weight")).alias("population_21plus"),
    )
)

# A few slivers of 2020 WA blocks sat in 2010 Idaho/Oregon blocks (boundary cleanup), and any 2020 block missing from the relationship file would show up as null
print("2020 blocks missing from relationship file:", allocated.filter(pl.col("cbg2010").is_null()).shape[0])
print("population allocated outside WA's 2010 boundaries:", allocated.filter(pl.col("STATE_2010") != STATE_FIPS)["population_total"].sum())

cbg2010_df = (allocated
    .filter(pl.col("STATE_2010") == STATE_FIPS)
    .group_by("cbg2010")
    .agg(
        pl.col("population_total").sum(),
        pl.col("population_21plus").sum(),
    )
    .with_columns(pl.col("cbg2010").str.slice(0, 5).alias("county_fips"))
    .select(["cbg2010", "county_fips", "population_total", "population_21plus"])
    .sort("cbg2010")
)

cbg2010_df.write_parquet("processed-data/3b_cbg2010_population_2020.parquet")

cbg2010_df"""))

cells.append(nbf.v4.new_code_cell("""### CHECKS
# (1) State total pulled straight from the API (the one number with no privacy noise in it)
state_total = get_census("state:" + STATE_FIPS)["population_total"].item()

print("state total:          ", state_total)
print("[3b] 2020 CBG sum:    ", cbg2020_df["population_total"].sum())
print("[3b'] 2010 CBG sum:   ", round(cbg2010_df["population_total"].sum(), 1))
print("[3c] county sum:      ", county_df["population_total"].sum())
print("[3a] ZCTA sum:        ", zcta_df["population_total"].sum(), "(ZCTAs don't quite cover everything, e.g. some large unpopulated areas)")
print("[3d] place sum:       ", place_df["population_total"].sum(), "(places don't cover unincorporated areas)")
print("[3d] place types:     ", place_df["place_type"].value_counts().sort("place_type").rows()) """))

cells.append(nbf.v4.new_code_cell("""# (2) Does every CBG Veraset uses show up in [3b']? (If Veraset were on 2020 boundaries this would fail badly)
veraset_cbgs = (pl.scan_parquet(
        "raw-data/veraset-visits-wa/**/*.parquet",
        hive_partitioning = True
    )
    .select(pl.col("census_block_group").unique())
    .collect(engine = "streaming")
)

veraset_cbgs_wa = veraset_cbgs.filter(pl.col("census_block_group").str.starts_with(STATE_FIPS))

print("distinct Veraset CBGs:", veraset_cbgs.shape[0], "| in WA:", veraset_cbgs_wa.shape[0])
print("matched to [3b'] (2010 CBGs):", veraset_cbgs_wa.join(cbg2010_df, left_on = "census_block_group", right_on = "cbg2010", how = "semi").shape[0])
print("matched to [3b] (2020 CBGs): ", veraset_cbgs_wa.join(cbg2020_df, left_on = "census_block_group", right_on = "cbg2020", how = "semi").shape[0])

# Veraset CBGs with no match in [3b']
veraset_cbgs_wa.join(cbg2010_df, left_on = "census_block_group", right_on = "cbg2010", how = "anti")"""))

cells.append(nbf.v4.new_code_cell("""# (3) Comparing [3c] against the county table from earlier weeks (the ChatGPT-generated one)
old_counties = pl.read_csv("raw-data/WA-county-lookup.csv").with_columns(pl.col("geoid").cast(pl.String).alias("county_fips"))

comparison = (county_df
    .join(old_counties.select(["county_fips", "population_2020"]), on = "county_fips", how = "left")
    .with_columns((pl.col("population_total") - pl.col("population_2020")).alias("difference"))
)

print("counties that disagree:", comparison.filter(pl.col("difference") != 0).shape[0])
comparison.filter(pl.col("difference") != 0)"""))

cells.append(nbf.v4.new_code_cell(""))

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, sys.argv[1])
