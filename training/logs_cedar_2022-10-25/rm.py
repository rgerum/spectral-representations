from pathlib import Path
import shutil

for p in Path().rglob("**/model_save_best"):
    print(p)
    shutil.rmtree(p)
