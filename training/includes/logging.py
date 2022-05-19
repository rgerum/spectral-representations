import os
from pathlib import Path

import numpy as np


def get_git_hash():
    import subprocess
    try:
        short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
        short_hash = str(short_hash, "utf-8").strip()
        return short_hash
    except subprocess.CalledProcessError:
        return ""


def get_git_hash_long():
    import subprocess
    try:
        short_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        short_hash = str(short_hash, "utf-8").strip()
        return short_hash
    except subprocess.CalledProcessError:
        return ""


def get_output_path(func, locals):
    import inspect
    def check(v):
        if getattr(v, "dtype", None) and v.dtype.kind == "f":
            return float(v)
        if getattr(v, "dtype", None) and v.dtype.kind == "i":
            return int(v)
        return v
    kwargs = {x: check(locals[x]) for x in inspect.signature(func).parameters}

    from datetime import datetime
    parts = [
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        get_git_hash(),
    ]
    parts.extend([str(k) + "=" + str(v) for k, v in kwargs.items() if k != "output"])

    output = Path(kwargs['output'])  # / (" ".join(parts))
    import yaml
    output.mkdir(parents=True, exist_ok=True)
    arguments = dict(datetime=parts[0], commit=parts[1], commitLong=get_git_hash_long(), run_dir=os.getcwd())
    arguments.update(kwargs)
    print("arguments", arguments)

    with open(output / "arguments.yaml", "w") as fp:
        yaml.dump(arguments, fp)
    print("OUTPUT_PATH=\"" + str(output) + "\"")

    import shutil
    shutil.copy(inspect.getfile(func), output)

    return output

