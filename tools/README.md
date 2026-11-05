# tools

Helper scripts for running the processing notebooks, building the store review page, and refreshing the write-up. Run everything from the project root.

## Run order

The notebooks in `Task 1/` and `Task 2/` are numbered in the order they have to run. The prefix is the run order, not the table label: `step1_` builds the [3] tables because `step2_` and `step3_` read them.

| run | notebook | builds |
|---|---|---|
| 1 | `Task 1/step1_geography_helpers_processing.ipynb` | [3a]–[3d] census population by ZCTA, CBG, county, place |
| 2 | `Task 1/step2_wslcb_safegraph_locations_processing.ipynb` | [1a]–[1e], [1e'] WSLCB stores, coordinates, approved store labels, and the irregularities log |
| 3 | `Task 1/step3_veraset_devices_processing.ipynb` | [2ai]/[2aii] device samples, [2b] home locations (slow: it scans all of the raw Veraset data) |
| 4 | `Task 1/step4_dispensary_visits_processing.ipynb` | [4ai]/[4aii] home geography, [4bi]/[4bii] cleaned dispensary visits, [4ci]–[4ciii'] summary tables and charts |
| 5 | `Task 1/step5_visits_and_access_processing.ipynb` | [5a] home-area visit shares, [5b] area store access, [5ci]/[5cii], [5d] visits by access |
| 6 | `Task 1/step6_store_coverage_processing.ipynb` | [6a]/[6b] storefront coverage, bar charts and maps (needs only `step1_` and `step2_`) |
| 7 | `Task 2/step7_device_flags_processing.ipynb` | Task 2, yearly sample only: [7a] retail worker, [7b] delivery/wholesale worker, [7c] high-frequency flags, [7d] device-quarter flags, [7e] visit summaries before/after exclusions. Charts go to `Task 2/Charts and Figures/` (needs `step4_`) |
| 8 | `Task 2/step8_device_activity_processing.ipynb` | Task 2, yearly sample, all Veraset visits (cleaned like [4bi]): [8a] total/home/non-home visits per device, [8bi]/[8bii] gaps between visits (within day start-to-start and end-to-start, across days), [8c] activity stability (daily/weekly visits and 1/gap), [8d] sample-level distributions. About 1.5 minutes, 9 GB of memory |

```bash
python3 tools/run_nb.py "Task 1/step1_geography_helpers_processing.ipynb"
python3 tools/run_nb.py "Task 1/step2_wslcb_safegraph_locations_processing.ipynb"
python3 tools/run_nb.py "Task 1/step3_veraset_devices_processing.ipynb"
python3 tools/run_nb.py "Task 1/step4_dispensary_visits_processing.ipynb"
python3 tools/run_nb.py "Task 1/step5_visits_and_access_processing.ipynb"
python3 tools/run_nb.py "Task 1/step6_store_coverage_processing.ipynb"
python3 tools/run_nb.py "Task 2/step7_device_flags_processing.ipynb"
python3 tools/run_nb.py "Task 2/step8_device_activity_processing.ipynb"
python3 tools/build_writeup.py
```

Charts from `step4_`–`step6_` are also saved to `Charts and Figures/` as .html files.

The notebooks read and write paths relative to the project root (`processed-data/`, `raw-data/`), so run them from the project root as above. `run_nb.py` uses the folder you run it from as the working directory. Opened directly in Jupyter, they would start in `Task 1/` instead, so change to the project root first (for example `%cd ..` in a scratch cell).

## Files

| file | what it does |
|---|---|
| `run_nb.py` | runs a notebook in place with nbclient and prints each cell's output |
| `build_writeup.py` | copies the current charts into `Results Write-Up.html` |
| `build_review_1e.py` | builds `review/1e_review_page.html` and `review/1e_review_items.csv` from [1e] |
| `build_match_audit.py` | builds `Task 1/WSLCB Dispensary Matching.xlsx` from [1a], [1e], [6a]: every approved WSLCB storefront to Veraset place link side by side, for checking by hand. Rebuilding overwrites the file, including any verdicts typed into it |
| `review_1e_template.html` | the page template `build_review_1e.py` fills in |

The notebook generator scripts (`make_step*.py`, `step2_cells.py`) are kept locally only and are gitignored. The line numbers in `Reproduction Guide.html` refer to them.

## Store review ([1e])

After running `step2_`, build the page with `python3 -W ignore tools/build_review_1e.py`, then serve it on a fixed port, since the browser keeps your verdicts per address (`localhost:8793`):

```bash
python3 -m http.server 8793 --bind 127.0.0.1 --directory review
```

Open http://localhost:8793/1e_review_page.html. Keys: 1 = yes, 2 = no, 3 = unsure, ←/→ to move, n = next unreviewed, / = search.
When done, export the verdicts and save them as `review/1e_review_verdicts.csv`. Rerunning `Task 1/step2_wslcb_safegraph_locations_processing.ipynb` picks them up.

## Write-up

`Results Write-Up.html` (project root) answers the sample-restriction questions; `Reproduction Guide.html` maps every paragraph, table, and figure in it to the code that produced it. The write-up's numbers are typed in, so update them by hand after changing anything. Its charts are copies of the ones in `Charts and Figures/` (with plotly.js) built into the page, so it works as a single file offline; refresh them after rerunning `step4_`/`step5_` with:

```bash
python3 tools/build_writeup.py
```
