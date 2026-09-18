"""
execute_notebook.py — run the committed notebook so GitHub shows real outputs.

WHY THIS DESIGN
---------------
A notebook committed without outputs shows a reviewer nothing until they run
it. Committing an executed copy fixes that, but a hand-run notebook drifts:
someone re-runs a cell, saves, and the file no longer matches the generator.
This script makes execution a single reproducible step: regenerate from
build_notebook.py, execute every cell in order with a fresh kernel, refuse to
proceed if any cell errored, validate the result, and stamp it with the date
and model so a reader knows exactly which run they are looking at.

It needs GOOGLE_API_KEY, because two cells call Gemini. It fails fast and
loudly if the key is absent rather than hanging on the notebook's key prompt.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbformat.v4 import new_markdown_cell

from build_notebook import OUTPUT, build
from langchain_context_chain import DEFAULT_MODEL_NAME

CELL_TIMEOUT_SECONDS = 600


def execute_in_place(path: Path) -> None:
    """Execute every cell with a fresh python3 kernel, writing outputs back into the same file."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            f"--ExecutePreprocessor.timeout={CELL_TIMEOUT_SECONDS}",
            "--ExecutePreprocessor.kernel_name=python3",
            str(path),
        ],
        check=True,
        cwd=path.parent,
    )


def verify_outputs(path: Path) -> int:
    """Every code cell must have output and none may hold an error. Returns the number of code cells."""
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        code_cells += 1
        outputs = cell.get("outputs", [])
        errors = [o for o in outputs if o.get("output_type") == "error"]
        if errors:
            raise RuntimeError(f"cell {index} raised {errors[0].get('ename')}: {errors[0].get('evalue')}")
        if not outputs:
            raise RuntimeError(f"cell {index} produced no output")
    return code_cells


def stamp(path: Path, executed_at: datetime) -> None:
    """Insert a dated note after the title so the baked-in outputs are labelled as one captured run."""
    notebook = nbformat.read(path, as_version=4)
    note = new_markdown_cell(
        f"> **This copy was executed on {executed_at:%Y-%m-%d %H:%M UTC} against `{DEFAULT_MODEL_NAME}`.** "
        "The outputs below come from that one run and will not update on their own. "
        "Run all cells yourself (here or in Colab via the badge above) to generate a fresh result."
    )
    notebook.cells.insert(1, note)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def main() -> int:
    if not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY is not set. Executing the notebook needs it; nothing was changed.")
        return 2

    build()  # start from a clean, validated, output-free notebook
    execute_in_place(OUTPUT)
    code_cells = verify_outputs(OUTPUT)
    executed_at = datetime.now(timezone.utc)
    stamp(OUTPUT, executed_at)
    print(f"Executed {code_cells} code cells with no errors; stamped {executed_at:%Y-%m-%d %H:%M UTC}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
