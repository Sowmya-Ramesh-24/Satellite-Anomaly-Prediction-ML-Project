
import subprocess, sys
from pathlib import Path

NB = Path(__file__).parent / "notebooks"
ORDER = ["lr"]

for name in ORDER:
    print(f"running {name}.ipynb ...", flush=True)
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=600", str(NB / f"{name}.ipynb")], check=True)
print("done -> Linear Regression results are in notebooks/lr.ipynb and results/")
