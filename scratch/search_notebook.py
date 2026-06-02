import json
import sys

# Reconfigure stdout to handle unicode properly
sys.stdout.reconfigure(encoding='utf-8')

notebook_path = r"d:\DH412-History-and-the-Digital\arabesque_analysis.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "fit_transform" in source or "PCA" in source:
            print(f"Cell Index: {idx}")
            print("--- Source ---")
            print(source)
            print("--------------\n")
