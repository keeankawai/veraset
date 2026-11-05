"""Puts the charts from "Charts and Figures/" straight into "Results Write-Up.html", so the page works on its own (no frames, no internet).

Each chart spot in the write-up is either an <iframe src="Charts%20and%20Figures/NAME.html"> (first run) or a block between
<!--chart:NAME--> and <!--/chart:NAME--> (later runs), and gets refilled from the current chart file. plotly.js goes in the page once.
Rerun after Steps 4/5 so the embedded charts match the latest run:  python3 tools/build_writeup.py
"""
import os, re
import plotly.offline

S = os.path.dirname(os.path.abspath(__file__))  # tools/
os.chdir(os.path.dirname(S))  # project root

PAGE = "Results Write-Up.html"
CHART_FOLDER = "Charts and Figures"
HEIGHT = "470px"

page = open(PAGE, encoding = "utf-8").read()

def chart_block(name):
    html = open(f"{CHART_FOLDER}/{name}.html", encoding = "utf-8").read()
    start = html.index('<div id="')  # the plot's div, then the script that draws it
    end = html.rindex("</script>") + len("</script>")
    body = html[start:end].replace("height:100%; width:100%;", f"height:{HEIGHT}; width:100%;", 1)
    return f'<!--chart:{name}--><div class="chartbox">{body}</div><!--/chart:{name}-->'

names = []
def refill(match):
    name = match.group(1) or match.group(2)
    names.append(name)
    return chart_block(name)

page = re.sub(r'<iframe[^>]*src="Charts%20and%20Figures/([^"]+)\.html"[^>]*></iframe>|<!--chart:([^>]+?)-->.*?<!--/chart:\2-->',
              refill, page, flags = re.S)

# plotly.js itself, once, in the head
plotly_js = f'<!--plotlyjs--><script type="text/javascript">window.PlotlyConfig = {{MathJaxConfig: "local"}};</script><script type="text/javascript">{plotly.offline.get_plotlyjs()}</script><!--/plotlyjs-->'
if "<!--plotlyjs-->" in page:
    page = re.sub(r"<!--plotlyjs-->.*?<!--/plotlyjs-->", lambda m: plotly_js, page, flags = re.S)
else:
    page = page.replace("</head>", plotly_js + "\n</head>", 1)

# the box the charts sit in (white, since the charts are drawn on white)
if ".chartbox" not in page:
    page = page.replace("</style>", ".chartbox { border: 1px solid var(--rule); background: #fff; border-radius: 4px; overflow: hidden; }\n</style>", 1)

open(PAGE, "w", encoding = "utf-8").write(page)
print("charts embedded:", len(names), "| page size:", round(os.path.getsize(PAGE) / 1e6, 1), "MB")
