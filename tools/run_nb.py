import nbformat, sys
from nbclient import NotebookClient
nb = nbformat.read(sys.argv[1], as_version=4)
try:
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
finally:
    nbformat.write(nb, sys.argv[1])
    for i, c in enumerate(nb.cells):
        for o in c.get("outputs", []):
            t = o.get("text") or o.get("data", {}).get("text/plain") or (o.get("ename", "") + ": " + o.get("evalue", ""))
            print(f"--- [{i}]", str(t)[:1500])
