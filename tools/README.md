# tools

Scripts that generate and run the processing notebooks, plus the Step 1 store review page. Run everything from the project root.

| file | what it does |
|---|---|
| `make_step1.py` | writes `wslcb_safegraph_locations_processing.ipynb` (Step 1, [1a]–[1e] and the irregularities log) |
| `make_step3.py` | writes `geography_helpers_processing.ipynb` (Step 3, [3a]–[3d]) |
| `make_step5.py` | writes `visits_and_access_processing.ipynb` (Step 5, [5a] home-area visit shares, [5b] area store access, [5ci]/[5cii] device visits + access, [5d] visits by access) |
| `make_step6.py` | writes `store_coverage_processing.ipynb` (Step 6, [6a] storefront coverage and reasons missing, [6b] matched vs. missing by county / place type / place size, bar charts and maps) |
| `make_step4.py` | writes `dispensary_visits_processing.ipynb` (Step 4, [4ai]/[4aii] home geography, device windows, [4bi]/[4bii] cleaned dispensary visits, [4ci]/[4ci'] sample sizes and representativeness, [4cii] visit distributions, [4ciii]/[4ciii'] start times and dwell, charts) |
| `make_step2.py` | writes `veraset_devices_processing.ipynb` (Step 2): a word-for-word copy of the existing notebook (the hand-written part plus the cells from `step2_cells.py`), so every line has a number. Not run yet; rerunning Step 2 is slow |
| `step2_cells.py` | the cells that were appended to `veraset_devices_processing.ipynb` (now also in `make_step2.py`) |
| `run_nb.py` | runs a notebook in place with nbclient and prints each cell's output |
| `build_writeup.py` | copies the current charts into `Results Write-Up.html` |
| `build_review_1e.py` | builds `review/1e_review_page.html` and `review/1e_review_items.csv` from [1e] |
| `review_1e_template.html` | the page template `build_review_1e.py` fills in |

## Rebuilding Step 1

```bash
python3 tools/make_step1.py wslcb_safegraph_locations_processing.ipynb && python3 tools/run_nb.py wslcb_safegraph_locations_processing.ipynb
python3 -W ignore tools/build_review_1e.py
```

## Rebuilding Step 4

Needs Steps 1–3 first. Charts also get saved to `Charts and Figures/` as .html files.

```bash
python3 tools/make_step4.py dispensary_visits_processing.ipynb && python3 tools/run_nb.py dispensary_visits_processing.ipynb
```

## Rebuilding Step 5

Needs Step 4 (and Step 1's [1e'] storefront months) first.

```bash
python3 tools/make_step5.py visits_and_access_processing.ipynb && python3 tools/run_nb.py visits_and_access_processing.ipynb
```

## Rebuilding Step 6

Needs Steps 1 and 3 (it reads [1a], [1d], [1e], [1e'], [3c], [3d], and the TIGER block group shapes Step 1 downloaded).

```bash
python3 tools/make_step6.py store_coverage_processing.ipynb && python3 tools/run_nb.py store_coverage_processing.ipynb
```

## Store review ([1e])

Serve the page on a fixed port, since the browser keeps your verdicts per address (`localhost:8793`):

```bash
python3 -m http.server 8793 --bind 127.0.0.1 --directory review
```

Open http://localhost:8793/1e_review_page.html. Keys: 1 = yes, 2 = no, 3 = unsure, ←/→ to move, n = next unreviewed, / = search.
When done, export the verdicts and save them as `review/1e_review_verdicts.csv`. Rerunning the Step 1 notebook picks them up.

## Write-up

`Results Write-Up.html` (project root) answers the sample-restriction questions; `Reproduction Guide.html` maps every paragraph, table, and figure in it to a notebook cell and a line in these scripts. Comments starting with `Write-up:` in the scripts point back to its sections. The write-up's numbers are typed in, so update them by hand after changing anything. Its charts are copies of the ones in `Charts and Figures/` (with plotly.js) built into the page, so it works as a single file offline; refresh them after rerunning Steps 4/5 with:

```bash
python3 tools/build_writeup.py
```
