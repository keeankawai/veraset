import nbformat as nbf
import sys

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""This document aims to accomplish the following:

(a) For every WSLCB storefront operating at some point in 2019-2021, whether we matched it to Veraset (an approved [1e] label in a month it was operating), how many of its operating months are covered, and, if it's missing, why. Label as ([6a])

(b) The geographic representativeness of the matched stores against all stores, and of the missing ones, by county, by type of place (city, town, CDP, or outside any place), and by city size. Label as ([6b])

(c) Bar charts of the above, the coverage month by month, and a map of Washington with a dot for every missing store

Notes:
- Stores are counted as storefronts (a WSLCB address), one of the two units in the Step 1 match rates (the other, licenses, is 435 of 546 matched, almost the same rate)
- "Operating" is the [1e'] rule: a license listed as operating there in the WSLCB sample before or after the month. Months only the undated current list vouches for don't count, so the universe is the 542 storefronts operating in some month of 2019-2021 by the dated samples
- Where a store is (county, place, ZCTA, coordinates) comes from geocoding its WSLCB address ([1d]), not from Veraset, so matched and missing stores are placed the same way
""")

code("""### IMPORTS AND PRELIMINARIES
## WORKHORSE LIBRARY
import polars as pl

## CHARTS (same as week 6), plus the map
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd # for county outlines, built from the TIGER block group shapes Step 1 downloaded
import json

## EXTRAS
from scipy.stats import chi2_contingency # is matching independent of where the store is?

# Cleans up some cell outputs
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)

STOREFRONT = ["street_address_norm", "city_norm"]
CHART_FOLDER = "Charts and Figures"

def show(fig, name):
    fig.show()
    fig.write_html(f"{CHART_FOLDER}/{name}.html", include_plotlyjs = "cdn") # so they can be opened outside the notebook too""")

code("""### PART A
## Write-up: section 7.1 (which stores are matched, and why the others are missing), table T7.1
## Output: one row per WSLCB storefront operating in some month of 2019-2021 (by the dated samples)
##         Columns are "street_address_norm", "city_norm", "tradenames", "licenses" (every license on the storefront's open months in [1e']; this can include a license only the current list has, so for license
##         counts use the Step 1 match rates, which are 546 licenses / 435 matched), "open_months",
##         "covered_months" (open months with an approved [1e] label), "share_months_covered", "matched" (covered_months > 0), "reason_missing" (see below,
##         null when matched), the store's location from [1d] ("latitude", "longitude", "geo_method", "county_fips", "county_name", "place_fips", "place_name",
##         "place_type", "zcta", "cbg2010"), "place_population_21plus" (from [3d]), "place_size" (by 21+ population, or "outside any place")
##         reason_missing, checked in this order:
##           "no Veraset place at the address"            - nothing in Veraset at the address (or a different-address candidate) in any month
##           "only other businesses at the address"       - Veraset has places at the address, but none goes by the store's name
##           "rejected in review"                         - the only possible matches got a "no" in the manual review
##           "name only outside its WSLCB months"         - a place with the store's name is there, but only in months WSLCB doesn't have a license operating
## Name: 6a_storefront_coverage.parquet
# (1) Open months, from [1e'], and which of them have an approved label in [1e]
open_months = (pl.read_parquet("processed-data/1e_storefront_months.parquet")
    .filter(~pl.col("current_list_only"))
)
labels = pl.read_parquet("processed-data/1e_storefront_place_months.parquet")
covered = labels.filter(pl.col("approved")).select(STOREFRONT + ["month"]).unique().with_columns(pl.lit(True).alias("covered"))

storefronts = (open_months
    .join(covered, on = STOREFRONT + ["month"], how = "left")
    .group_by(STOREFRONT)
    .agg(
        pl.col("operating_licenses").explode().unique().sort().alias("licenses"),
        pl.len().alias("open_months"),
        pl.col("covered").fill_null(False).sum().alias("covered_months"),
    )
    .with_columns(
        (pl.col("covered_months") / pl.col("open_months")).alias("share_months_covered"),
        (pl.col("covered_months") > 0).alias("matched"),
    )
)

# (2) Why the missing ones are missing, from what [1e] has at (or near) their address in any month
label_summary = (labels
    .group_by(STOREFRONT)
    .agg(
        pl.col("name_check").is_in(["auto", "yes_in_review"]).any().alias("has_store_name"),
        pl.col("name_check").is_in(["review", "no_in_review"]).any().alias("had_review_item"),
        pl.col("tradenames").first(),
    )
)
storefronts = (storefronts
    .join(label_summary, on = STOREFRONT, how = "left")
    .with_columns(
        pl.when(pl.col("matched")).then(None)
          .when(pl.col("has_store_name").is_null()).then(pl.lit("no Veraset place at the address"))
          .when(pl.col("has_store_name")).then(pl.lit("name only outside its WSLCB months"))
          .when(pl.col("had_review_item")).then(pl.lit("rejected in review"))
          .otherwise(pl.lit("only other businesses at the address"))
          .alias("reason_missing")
    )
)

# Storefronts [1e] never saw still need their tradenames, from [1a]
tradenames = (pl.read_parquet("processed-data/1a_wslcb_locations.parquet")
    .group_by(STOREFRONT)
    .agg(pl.col("tradenames").explode().unique().sort().alias("all_tradenames"))
)

# (3) Where each storefront is, from [1d] (a storefront has a row per license there, all with the same coordinates), and the size of its place
store_locations = (pl.read_parquet("processed-data/1d_wslcb_location_coordinates.parquet")
    .sort("license")
    .group_by(STOREFRONT, maintain_order = True)
    .agg(pl.col(["latitude", "longitude", "geo_method", "county_fips", "county_name", "place_fips", "place_name", "place_type", "zcta", "cbg2010"]).first())
)
place_sizes = pl.read_parquet("processed-data/3d_place_population_2020.parquet").select(["place_fips", pl.col("population_21plus").alias("place_population_21plus")])

storefront_coverage = (storefronts
    .join(tradenames, on = STOREFRONT, how = "left")
    .join(store_locations, on = STOREFRONT, how = "left")
    .join(place_sizes, on = "place_fips", how = "left")
    .with_columns(
        pl.coalesce("tradenames", "all_tradenames").alias("tradenames"),
        pl.when(pl.col("place_fips").is_null()).then(pl.lit("outside any place"))
          .when(pl.col("place_population_21plus") < 10000).then(pl.lit("under 10k adults"))
          .when(pl.col("place_population_21plus") < 50000).then(pl.lit("10k-50k adults"))
          .when(pl.col("place_population_21plus") < 100000).then(pl.lit("50k-100k adults"))
          .otherwise(pl.lit("100k+ adults"))
          .alias("place_size"),
        pl.col("place_type").fill_null("outside any place"),
    )
    .select(STOREFRONT + ["tradenames", "licenses", "open_months", "covered_months", "share_months_covered", "matched", "reason_missing",
                          "latitude", "longitude", "geo_method", "county_fips", "county_name", "place_fips", "place_name", "place_type", "zcta", "cbg2010",
                          "place_population_21plus", "place_size"])
    .sort(STOREFRONT)
)
storefront_coverage.write_parquet("processed-data/6a_storefront_coverage.parquet")

print("storefronts:", storefront_coverage.shape[0], "| matched:", storefront_coverage["matched"].sum(), f"({storefront_coverage['matched'].mean():.1%})",
      "| open months covered:", f"{storefront_coverage['covered_months'].sum() / storefront_coverage['open_months'].sum():.1%}",
      "| without coordinates:", storefront_coverage["latitude"].null_count())

storefront_coverage.group_by("reason_missing").agg(pl.len().alias("storefronts"), pl.col("open_months").sum()).sort("storefronts", descending = True)""")

code("""# (2) How much of a matched store's time is covered: the share of its open months with an approved label
(storefront_coverage
    .filter(pl.col("matched"))
    .select(
        pl.len().alias("matched_storefronts"),
        pl.col("share_months_covered").mean().alias("mean"),
        pl.col("share_months_covered").median().alias("median"),
        (pl.col("share_months_covered") == 1).mean().alias("share_fully_covered"),
        (pl.col("share_months_covered") < 0.5).mean().alias("share_under_half"),
    ))""")

code("""### PART B
## Write-up: section 7.2 (representativeness), tables T7.2-T7.3
## Output: one row per grouping and group, comparing matched and missing storefronts with all of them
##         Groupings are "county", "place type" (city, town, CDP, outside any place), and "place size" (the place's 21+ population)
##         Columns are "grouping", "group", "storefronts", "matched", "missing", "match_rate", "share_of_all", "share_of_matched", "share_of_missing" (shares
##         of the statewide totals), "month_coverage" (covered months / open months), and "share_of_adults" (the group's share of WA's 21+ population;
##         for places, of the 21+ population living in that kind of place)
## Name: 6b_coverage_by_area.parquet
counties = pl.read_parquet("processed-data/3c_county_population_2020.parquet")
places = pl.read_parquet("processed-data/3d_place_population_2020.parquet")
state_adults = counties["population_21plus"].sum()

# 21+ population by place type and place size, with everyone outside a place as its own group
place_groups = places.with_columns(
    pl.when(pl.col("population_21plus") < 10000).then(pl.lit("under 10k adults"))
      .when(pl.col("population_21plus") < 50000).then(pl.lit("10k-50k adults"))
      .when(pl.col("population_21plus") < 100000).then(pl.lit("50k-100k adults"))
      .otherwise(pl.lit("100k+ adults"))
      .alias("place_size")
)
outside_adults = state_adults - places["population_21plus"].sum()
adults = {
    "county": counties.select(pl.col("county_name").alias("group"), "population_21plus"),
    "place type": pl.concat([place_groups.group_by(pl.col("place_type").alias("group")).agg(pl.col("population_21plus").sum()),
                             pl.DataFrame({"group": ["outside any place"], "population_21plus": [outside_adults]})], how = "vertical_relaxed"),
    "place size": pl.concat([place_groups.group_by(pl.col("place_size").alias("group")).agg(pl.col("population_21plus").sum()),
                             pl.DataFrame({"group": ["outside any place"], "population_21plus": [outside_adults]})], how = "vertical_relaxed"),
}

GROUPINGS = [("county", "county_name"), ("place type", "place_type"), ("place size", "place_size")]
results = []
tests = []
for grouping, column in GROUPINGS:
    by_group = (storefront_coverage
        .group_by(pl.col(column).alias("group"))
        .agg(
            pl.len().alias("storefronts"),
            pl.col("matched").sum().alias("matched"),
            (~pl.col("matched")).sum().alias("missing"),
            (pl.col("covered_months").sum() / pl.col("open_months").sum()).alias("month_coverage"),
        )
        .join(adults[grouping], on = "group", how = "full", coalesce = True) # keeps groups with adults but no stores (Garfield County)
        .with_columns(pl.col(["storefronts", "matched", "missing"]).fill_null(0))
        .with_columns(
            pl.lit(grouping).alias("grouping"),
            (pl.col("matched") / pl.col("storefronts")).alias("match_rate"),
            (pl.col("storefronts") / pl.col("storefronts").sum()).alias("share_of_all"),
            (pl.col("matched") / pl.col("matched").sum()).alias("share_of_matched"),
            (pl.col("missing") / pl.col("missing").sum()).alias("share_of_missing"),
            (pl.col("population_21plus") / pl.col("population_21plus").sum()).alias("share_of_adults"),
        )
        .sort("storefronts", descending = True)
    )
    results.append(by_group)

    # Summaries: how far the matched stores' distribution is from all stores' (dissimilarity, like Step 4), and a chi-square test of whether being matched
    # depends on the group. With 39 counties a lot of the cells are small, so the county test is only a rough guide
    with_stores = by_group.filter(pl.col("storefronts") > 0)
    chi2, p_value, dof, expected = chi2_contingency(with_stores.select(["matched", "missing"]).to_numpy())
    tests.append({
        "grouping": grouping,
        "groups_with_stores": with_stores.shape[0],
        "dissimilarity_matched_vs_all": ((by_group["share_of_matched"] - by_group["share_of_all"]).abs().sum() / 2),
        "dissimilarity_missing_vs_all": ((by_group["share_of_missing"] - by_group["share_of_all"]).abs().sum() / 2),
        "chi2": chi2, "dof": dof, "p_value": p_value,
        "share_of_cells_expected_under_5": (expected < 5).mean(),
    })

coverage_by_area = (pl.concat(results)
    .select(["grouping", "group", "storefronts", "matched", "missing", "match_rate", "share_of_all", "share_of_matched", "share_of_missing", "month_coverage",
             "share_of_adults"])
)
coverage_by_area.write_parquet("processed-data/6b_coverage_by_area.parquet")

coverage_tests = pl.DataFrame(tests)
coverage_tests.with_columns(pl.col(pl.Float64).round(3))""")

code("""# (2) Place type and size
coverage_by_area.filter(pl.col("grouping") != "county").with_columns(pl.col(pl.Float64).round(3))""")

code("""# (3) Counties
coverage_by_area.filter(pl.col("grouping") == "county").with_columns(pl.col(pl.Float64).round(3))""")

code("""# (4) Does coverage change over time? Share of open storefront-months with an approved label, by year
(open_months
    .join(covered, on = STOREFRONT + ["month"], how = "left")
    .group_by(pl.col("month").str.slice(0, 4).alias("year"))
    .agg(pl.len().alias("open_storefront_months"), pl.col("covered").fill_null(False).mean().alias("share_covered"))
    .sort("year"))""")

code("""### PART C, SUBPART I: BAR CHARTS
## Write-up: section 7, figures F7.1-F7.6
GROUP_ORDER = {
    "place type": ["city", "town", "CDP", "outside any place"],
    "place size": ["100k+ adults", "50k-100k adults", "10k-50k adults", "under 10k adults", "outside any place"],
}

# (1) Match rate by county, biggest counties (by number of stores) first, with the number of stores on each bar
county_df = coverage_by_area.filter((pl.col("grouping") == "county") & (pl.col("storefronts") > 0)).sort("storefronts", descending = True)

fig1 = px.bar(
    county_df.with_columns((pl.col("matched").cast(pl.String) + "/" + pl.col("storefronts").cast(pl.String)).alias("label")).to_pandas(),
    x="group",
    y="match_rate",
    text="label",
    labels={
        "group": "County (most stores first)",
        "match_rate": "Share of storefronts matched",
    },
    title="Share of WSLCB Storefronts Matched to Veraset, by County (matched/all on each bar)",
)

fig1.update_xaxes(
    tickangle=45
)
fig1.update_traces(textposition="outside", textangle=0, textfont_size=9)
fig1.add_hline(y=storefront_coverage["matched"].mean(), line_dash="dot", annotation_text="statewide")

show(fig1, "store_match_rate_by_county")

# (2) Share of all stores, matched stores, and missing stores in each county (the same comparison as the week 6 representativeness chart)
plot_df = (county_df
    .unpivot(index="group", on=["share_of_all", "share_of_matched", "share_of_missing", "share_of_adults"], variable_name="source", value_name="share")
    .with_columns(pl.col("source").replace({
        "share_of_all": "All storefronts",
        "share_of_matched": "Matched storefronts",
        "share_of_missing": "Missing storefronts",
        "share_of_adults": "2020 Census, adults 21+",
    }))
)

fig2 = px.bar(
    plot_df.to_pandas(),
    x="group",
    y="share",
    color="source",
    barmode="group",
    labels={
        "group": "County",
        "share": "Share of the statewide total",
        "source": "",
    },
    title="Where the Matched and Missing Stores Are, by County",
)

fig2.update_xaxes(
    tickangle=45
)

show(fig2, "store_coverage_shares_by_county")

# (3) Match rate by type and size of place
for grouping in ["place type", "place size"]:
    group_df = (coverage_by_area
        .filter(pl.col("grouping") == grouping)
        .with_columns(
            pl.col("group").replace_strict({g: i for i, g in enumerate(GROUP_ORDER[grouping])}).alias("order"),
            (pl.col("matched").cast(pl.String) + "/" + pl.col("storefronts").cast(pl.String)).alias("label"),
        )
        .sort("order")
    )

    fig3 = px.bar(
        group_df.to_pandas(),
        x="group",
        y="match_rate",
        text="label",
        labels={
            "group": "Place type" if grouping == "place type" else "Size of the place the store is in",
            "match_rate": "Share of storefronts matched",
        },
        title=f"Share of WSLCB Storefronts Matched to Veraset, by {grouping.title()} (matched/all on each bar)",
    )

    show(fig3, f"store_match_rate_by_{grouping.replace(' ', '_')}")

# (4) Why the missing stores are missing
reasons = storefront_coverage.filter(~pl.col("matched")).group_by("reason_missing").len().sort("len", descending = True)

fig4 = px.bar(
    reasons.to_pandas(),
    x="reason_missing",
    y="len",
    text="len",
    labels={
        "reason_missing": "",
        "len": "Missing storefronts",
    },
    title="Why Storefronts Are Missing",
)

show(fig4, "store_missing_reasons")

# (5) Coverage month by month: the share of open storefronts with an approved label
monthly = (open_months
    .join(covered, on = STOREFRONT + ["month"], how = "left")
    .group_by("month")
    .agg(pl.len().alias("open_storefronts"), pl.col("covered").fill_null(False).mean().alias("share_covered"))
    .sort("month")
)

fig5 = px.line(
    monthly.to_pandas(),
    x="month",
    y="share_covered",
    labels={
        "month": "Month",
        "share_covered": "Share of open storefronts matched that month",
    },
    title="Share of Open WSLCB Storefronts with an Approved Veraset Label, by Month",
)

fig5.update_yaxes(range=[0, 1])

show(fig5, "store_coverage_by_month")""")

code("""### PART C, SUBPART II: MAP
## Write-up: section 7, figures F7.7-F7.8
# County outlines: the 2020 block group shapes Step 1 downloaded, merged by county and simplified (so the map file stays small)
bg2020_shapes = gpd.read_file("zip://raw-data/census/tl_2020_53_bg.zip")[["STATEFP", "COUNTYFP", "geometry"]]
county_shapes = (bg2020_shapes
    .assign(county_fips = bg2020_shapes["STATEFP"] + bg2020_shapes["COUNTYFP"])
    .dissolve(by = "county_fips")
    .reset_index()[["county_fips", "geometry"]]
    .to_crs("EPSG:4326")
)
county_shapes["geometry"] = county_shapes.geometry.simplify(0.003)
county_geojson = json.loads(county_shapes.to_json())

map_counties = (counties.select(["county_fips", "county_name"])
    .join(coverage_by_area.filter(pl.col("grouping") == "county").select([pl.col("group").alias("county_name"), "storefronts", "matched", "match_rate"]),
          on = "county_name", how = "left")
)

# Counties shaded by their match rate (Garfield, with no store, stays blank)
fig6 = px.choropleth(
    map_counties.to_pandas(),
    geojson=county_geojson,
    locations="county_fips",
    featureidkey="properties.county_fips",
    color="match_rate",
    color_continuous_scale="Blues",
    range_color=(0, 1),
    hover_name="county_name",
    hover_data={"county_fips": False, "storefronts": True, "matched": True, "match_rate": ":.0%"},
    labels={"match_rate": "Share matched"},
)

# A dot per store: matched ones small and grey, missing ones red
fig6.update_traces(marker_opacity=0.55, marker_line_color="white", selector=dict(type="choropleth")) # light enough for the dots to show

for matched, name, color, size in [(True, "Matched storefront", "#2ca02c", 6), (False, "Missing storefront", "#d62728", 6)]:
    dots = (storefront_coverage
        .filter((pl.col("matched") == matched) & pl.col("latitude").is_not_null())
        .with_columns((pl.col("tradenames").list.join(" / ") + "<br>" + pl.col("street_address_norm") + ", " + pl.col("city_norm")
                       + pl.when(pl.col("matched")).then(pl.lit("")).otherwise("<br>" + pl.col("reason_missing"))).alias("hover"))
    )
    fig6.add_trace(go.Scattergeo(
        lon=dots["longitude"].to_list(),
        lat=dots["latitude"].to_list(),
        mode="markers",
        marker=dict(size=size, color=color, line=dict(width=0.5, color="white")),
        name=name,
        text=dots["hover"].to_list(),
        hoverinfo="text",
    ))

fig6.update_layout(
    legend=dict(orientation="h", y=0),
    height=560,
    margin={"r": 0, "t": 60, "l": 0, "b": 0},
)

# Once for the whole state, and once zoomed in on Puget Sound, where most of the stores are
for map_name, title, bounds in [
    ("store_coverage_map", "WSLCB Storefronts Operating in 2019-2021: Matched (green) and Missing (red)", None),
    ("store_coverage_map_puget_sound", "The Same, Zoomed in on Puget Sound", dict(lonaxis_range=[-123.3, -121.6], lataxis_range=[46.85, 48.35])),
]:
    if bounds is None:
        fig6.update_geos(fitbounds="locations", visible=False)
    else:
        fig6.update_geos(fitbounds=False, visible=False, projection_type="mercator", **bounds)
    fig6.update_layout(title=title)

    show(fig6, map_name)""")

md("""### NOTES
- 427 of the 542 storefronts operating in 2019-2021 are matched (78.8%), covering 61.7% of their open months. Of the 115 missing: 63 have no Veraset place at the address at all, 38 only have other businesses there, 10 have a place with the store's name but only outside the months WSLCB has a license operating, and 4 were rejected in the manual review.
- Matched stores aren't covered for their whole run: on average 74% of their open months (median 83%); 39% are covered in every open month and 22% in fewer than half. Coverage rises over time (53% of open storefront-months in 2019, 65% in 2020, 67% in 2021).
- Type and size of place make no difference: 78% of stores in cities, 81% in CDPs, 79% outside any place, and 75-82% across city sizes (chi-square p = 0.89 and 0.78).
- Counties do differ (chi-square p = 0.01, rough, since most counties have few stores). Well covered: Spokane 31/33, Kitsap 19/21, Thurston 23/26, Whatcom 24/28. Poorly covered: Okanogan 5/13, Lewis 3/8, Benton 4/7, Stevens 6/10, Clark 14/22, Yakima 7/11, and Franklin's only store. King is about average for stores (93/124) but low for months (57%). The matched stores' county distribution is 6.3% away from all stores' (dissimilarity), the missing stores' 23.2%.
- Clark County is short on both counts: Portland's stores aren't in WA-only data, and 8 of its 22 stores aren't matched either, so its low visiting rate in Step 5 is partly coverage.
""")

code("")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, sys.argv[1])
