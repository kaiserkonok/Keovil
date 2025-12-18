from pathlib import Path
from typing import List, Tuple


TABULAR_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
}

# folders we never want to index
IGNORE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "node_modules",
    ".cache",
}


def collect_tabular_files(
    root_dir: Path,
) -> Tuple[List[Path], List[Path]]:
    """
    Recursively collect tabular files from a directory.

    Returns:
        (csv_files, excel_files)
    """

    if not root_dir.exists():
        raise FileNotFoundError(f"{root_dir} does not exist")

    csv_files: List[Path] = []
    excel_files: List[Path] = []

    for path in root_dir.rglob("*"):
        # skip directories
        if path.is_dir():
            continue

        # skip hidden files
        if path.name.startswith("."):
            continue

        # skip ignored directories
        if any(parent.name in IGNORE_DIRS for parent in path.parents):
            continue

        ext = path.suffix.lower()

        if ext == ".csv":
            csv_files.append(path)

        elif ext in {".xlsx", ".xls"}:
            excel_files.append(path)

    return csv_files, excel_files


# ---------------------------
# Terminal test
# ---------------------------
if __name__ == "__main__":
    base = Path("src/test_data")

    csvs, excels = collect_tabular_files(base)

    print("\nCSV files:")
    for f in csvs:
        print("-", f)

    print("\nExcel files:")
    for f in excels:
        print("-", f)
