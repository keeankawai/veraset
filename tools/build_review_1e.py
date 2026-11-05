"""Builds review/1e_review_page.html and review/1e_review_items.csv from processed-data/1e_storefront_place_months.parquet (Step 1, Part E)."""
import json, os, re, sys, urllib.parse
import polars as pl

S = os.path.dirname(os.path.abspath(__file__))  # tools/
os.chdir(os.path.dirname(S))  # project root, so the relative data paths work from anywhere

e = pl.read_parquet("processed-data/1e_storefront_place_months.parquet")
snap = pl.read_parquet("processed-data/1a_wslcb_snapshots_long.parquet")
loc = pl.read_parquet("processed-data/1a_wslcb_locations.parquet")
vpm = pl.read_parquet("processed-data/veraset_poi_months.parquet")

nb_cells = ["".join(c["source"]) for c in json.load(open("wslcb_safegraph_locations_processing.ipynb"))["cells"] if c["cell_type"] == "code"]
nb_src = next(c for c in nb_cells if "def normalize_address" in c)
exec(nb_src)  # normalize_text / normalize_address, so the WSLCB rows can be tied back to storefronts the same way as the notebook

KEY = ["safegraph_place_id", "location_name", "street_address_norm", "city_norm"]
STORE = ["street_address_norm", "city_norm"]
MONTHS = sorted(e["month"].unique().to_list())

def gmaps(q):
    return "https://www.google.com/maps/search/?api=1&query=" + urllib.parse.quote(q)

def gsearch(q):
    return "https://www.google.com/search?q=" + urllib.parse.quote(q)

def runs(rows, keys):
    # collapse consecutive months with the same values into "from -> to" runs
    out = []
    for r in rows:
        if out and all(out[-1][k] == r[k] for k in keys) and MONTHS.index(r["month"]) == MONTHS.index(out[-1]["to"]) + 1:
            out[-1]["to"] = r["month"]; out[-1]["months"] += 1; out[-1]["visits"] += r["visits"]
        else:
            out.append({**{k: r[k] for k in keys}, "from": r["month"], "to": r["month"], "months": 1, "visits": r["visits"]})
    return out

# ---- review items: one per place ID + label name + storefront
rev = e.filter(pl.col("name_check").is_in(["review", "yes_in_review", "no_in_review"]))
items_df = (rev.group_by(KEY)
    .agg(
        pl.col("placekey").drop_nulls().first(),
        pl.col("street_address").mode().first().alias("v_addr"),
        pl.col("city").mode().first().alias("v_city"),
        pl.col("naics_code").mode().first(),
        pl.col("name_rule").first(), pl.col("address_match").first(), pl.col("name_overlap").max(),
        pl.col("tradenames").first(),
        pl.col("num_visits").sum().alias("visits_all"),
        pl.col("num_visits").filter(pl.col("wslcb_operating")).sum().alias("visits_operating"),
        pl.col("wslcb_operating").sum().alias("months_operating"), pl.len().alias("months_seen"),
        pl.col("outline_evidence").first(), pl.col("outline_status").first(), pl.col("other_businesses_on_outline").first(),
        pl.col("parent_placekey").first(),
    )
    .sort("visits_operating", descending = True) # biggest first
)

approved_months = e.filter(pl.col("approved")).group_by(STORE).agg(pl.col("month").n_unique().alias("approved_months"))
store_op_months = e.filter(pl.col("wslcb_operating")).select(STORE + ["month"]).unique().group_by(STORE).agg(pl.len().alias("op_months_seen"))

# WSLCB rows per storefront
snap_n = snap.with_columns(normalize_address(pl.col("street_address")).alias("street_address_norm"), normalize_text(pl.col("city")).alias("city_norm"))
loc_n = loc.select(["license", "street_address_norm", "city_norm", "street_address", "zip5", "county", "first_operating", "last_operating", "tradenames", "statuses", "current_list_statuses"])

items = []
for r in items_df.iter_rows(named = True):
    sf = (r["street_address_norm"], r["city_norm"])
    lic_rows = loc_n.filter((pl.col("street_address_norm") == sf[0]) & (pl.col("city_norm") == sf[1]))
    licenses = []
    for lr in lic_rows.iter_rows(named = True):
        s = snap_n.filter((pl.col("license") == lr["license"])).sort("snapshot_date", nulls_last = True)
        licenses.append({
            "license": lr["license"], "tradenames": lr["tradenames"],
            "first_op": str(lr["first_operating"]) if lr["first_operating"] else None, "last_op": str(lr["last_operating"]) if lr["last_operating"] else None,
            "current": lr["current_list_statuses"],
            "snapshots": [{"date": str(x["snapshot_date"]) if x["snapshot_date"] else "current list", "tradename": x["tradename"], "status": x["status"],
                           "addr": x["street_address"], "city": x["city"], "here": (x["street_address_norm"], x["city_norm"]) == sf} for x in s.iter_rows(named = True)],
        })
    wslcb_addr = lic_rows["street_address"].mode().to_list()[0] if lic_rows.shape[0] else sf[0]
    zip5 = lic_rows["zip5"].drop_nulls().mode().to_list()[0] if lic_rows["zip5"].drop_nulls().shape[0] else ""

    # month by month for this exact label at this storefront
    months = (e.filter((pl.col("safegraph_place_id") == r["safegraph_place_id"]) & (pl.col("location_name") == r["location_name"])
                       & (pl.col("street_address_norm") == sf[0]) & (pl.col("city_norm") == sf[1]))
              .group_by("month").agg(pl.col("num_visits").sum().alias("visits"), pl.col("wslcb_operating").any().alias("op")).sort("month"))
    month_map = {m["month"]: m for m in months.iter_rows(named = True)}

    # everything this place ID has been called, any address
    hist = (vpm.filter(pl.col("safegraph_place_id") == r["safegraph_place_id"])
            .group_by(["month", "location_name", "street_address", "city"]).agg(pl.col("num_visits").sum().alias("visits"), pl.col("naics_code").mode().first())
            .sort(["month", "visits"], descending = [False, True]).unique("month", keep = "first", maintain_order = True)
            .rename({"location_name": "name", "street_address": "addr", "naics_code": "naics"}))
    history = runs(hist.to_dicts(), ["name", "addr", "city"])

    # other places Veraset has at the storefront's exact address
    others = (e.filter((pl.col("street_address_norm") == sf[0]) & (pl.col("city_norm") == sf[1]) & pl.col("address_match").is_in(["exact", "same_address_and_zip"])
                       & ~((pl.col("safegraph_place_id") == r["safegraph_place_id"]) & (pl.col("location_name") == r["location_name"])))
              .group_by(["safegraph_place_id", "location_name"])
              .agg(pl.col("street_address").first(), pl.col("month").min().alias("from"), pl.col("month").max().alias("to"), pl.col("num_visits").sum().alias("visits"),
                   pl.col("name_check").first(), pl.col("approved").any(), pl.col("outline_evidence").first())
              .sort("visits", descending = True))

    flags = []
    if r["address_match"] == "different_address":
        flags.append({"tag": "different address", "w": 3, "why": f"Veraset has it at {r['v_addr']}, {r['v_city']}; WSLCB has {wslcb_addr}, {sf[1].title()}. Is it the same storefront?"})
    if r["address_match"] == "same_address_and_zip":
        flags.append({"tag": "city named differently", "w": 0, "why": f"Same street address and zip as WSLCB; Veraset just calls the city \"{r['v_city']}\" instead of {sf[1].title()}"})
    if r["name_rule"] == "tradename":
        flags.append({"tag": "tradename + other words", "w": 2, "why": f"\"{r['location_name']}\" shares the tradename's distinctive words but has other words too (or smoke-shop words like GLASS/VAPE)"})
    elif r["name_rule"] == "cannabis_name":
        flags.append({"tag": "cannabis name, not the tradename", "w": 2, "why": "Looks like a cannabis store but shares no distinctive words with any WSLCB tradename at this address (renamed? the smoke shop next door?)"})
    elif r["name_rule"] == "category_only":
        flags.append({"tag": "424210 only", "w": 3, "why": "Neither name rule matches; it's only here because Safegraph files it under 424210 (where most dispensaries go)"})
    if r["outline_evidence"] in ("shared", "likely_shared"):
        others_txt = ", ".join((r["other_businesses_on_outline"] or [])[:8]) or "something at this address (2026 stand-in)"
        flags.append({"tag": "outline " + r["outline_evidence"].replace("_", " "), "w": 1, "why": f"The building outline is shared with: {others_txt} (kept as a flag, doesn't block approval)"})
    if r["months_operating"] < r["months_seen"]:
        flags.append({"tag": "outside WSLCB window", "w": 1, "why": f"Only {r['months_operating']} of the {r['months_seen']} months this label shows up are months WSLCB has a license operating here (only those can count)"})
    am = approved_months.filter((pl.col("street_address_norm") == sf[0]) & (pl.col("city_norm") == sf[1]))
    n_appr = am["approved_months"][0] if am.shape[0] else 0
    flags.append({"tag": "storefront already covered" if n_appr else "storefront has nothing approved yet", "w": 0 if n_appr else 2,
                  "why": f"This storefront has an approved place in {n_appr} month(s) already" if n_appr else "A yes here would add a new storefront to the matched set"})

    items.append({
        "id": "||".join([r["safegraph_place_id"], r["location_name"], sf[0], sf[1]]),
        "sgid": r["safegraph_place_id"], "placekey": r["placekey"], "name": r["location_name"], "v_addr": r["v_addr"], "v_city": r["v_city"],
        "naics": r["naics_code"], "name_rule": r["name_rule"], "address_match": r["address_match"],
        "overlap": r["name_overlap"], "tradenames": r["tradenames"], "store_addr": sf[0], "store_city": sf[1], "wslcb_addr": wslcb_addr, "zip": zip5,
        "visits": r["visits_operating"], "visits_all": r["visits_all"], "months_operating": r["months_operating"], "months_seen": r["months_seen"],
        "outline": r["outline_evidence"], "outline_status": r["outline_status"], "outline_others": r["other_businesses_on_outline"] or [], "parent": r["parent_placekey"],
        "new_store": n_appr == 0, "flags": flags,
        "grid": [[m, month_map[m]["visits"], month_map[m]["op"]] if m in month_map else [m, None, None] for m in MONTHS],
        "history": history, "licenses": licenses,
        "others": [{"name": o["location_name"], "addr": o["street_address"], "from": o["from"], "to": o["to"], "visits": o["visits"], "check": o["name_check"],
                    "approved": o["approved"], "outline": o["outline_evidence"]} for o in others.head(25).iter_rows(named = True)],
        "num_others": others.shape[0],
        "links": {"wslcb": gmaps(f"{wslcb_addr}, {sf[1].title()}, WA {zip5}"), "poi": gmaps(f"{r['location_name']}, {r['v_addr']}, {r['v_city']}, WA"),
                  "search": gsearch(f"{(r['tradenames'] or [''])[0]} {sf[1].title()} WA cannabis")},
    })

pl.DataFrame([{
    "id": it["id"], "safegraph_place_id": it["sgid"], "location_name": it["name"], "street_address_norm": it["store_addr"], "city_norm": it["store_city"],
    "veraset_address": it["v_addr"], "veraset_city": it["v_city"], "placekey": it["placekey"], "tradenames": " | ".join(it["tradenames"] or []),
    "name_rule": it["name_rule"], "address_match": it["address_match"], "outline_evidence": it["outline"], "visits_while_operating": it["visits"],
} for it in items]).write_csv("review/1e_review_items.csv")

payload = json.dumps({"items": items}, default = str).replace("</", "<\\/")
tpl = open(f"{S}/review_1e_template.html").read()
open("review/1e_review_page.html", "w").write(tpl.replace("__DATA__", payload))
print("items:", len(items), "| by reason:", items_df.group_by(["address_match", "name_rule"]).len().sort("len", descending = True).rows(),
      "| page:", os.path.getsize("review/1e_review_page.html") // 1024, "KB")
