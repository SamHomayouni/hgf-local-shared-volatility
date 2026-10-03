from pathlib import Path
from collections import defaultdict

ROOT = Path(".").resolve()

# Ignore obvious environment/cache directories while scanning
IGNORE_DIRS = {
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
}

files = []
dir_sizes = defaultdict(int)
extension_sizes = defaultdict(int)
extension_counts = defaultdict(int)

for path in ROOT.rglob("*"):
    if any(part in IGNORE_DIRS for part in path.parts):
        continue

    if not path.is_file():
        continue

    try:
        size = path.stat().st_size
    except OSError:
        continue

    relative = path.relative_to(ROOT)

    files.append((size, relative))

    # Add size to every parent directory
    parent = relative.parent
    while str(parent) != ".":
        dir_sizes[parent] += size
        parent = parent.parent

    suffix = path.suffix.lower() or "[no extension]"
    extension_sizes[suffix] += size
    extension_counts[suffix] += 1


def human_size(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


with open("repository_inventory.txt", "w", encoding="utf-8") as f:

    f.write("REPOSITORY INVENTORY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Repository root: {ROOT}\n\n")

    # --------------------------------------------------
    # Directory tree
    # --------------------------------------------------
    f.write("DIRECTORY / FILE STRUCTURE\n")
    f.write("=" * 80 + "\n")

    paths = sorted(
        [
            p.relative_to(ROOT)
            for p in ROOT.rglob("*")
            if not any(part in IGNORE_DIRS for part in p.parts)
        ],
        key=lambda p: str(p).lower(),
    )

    for p in paths:
        depth = len(p.parts) - 1
        indent = "    " * depth

        if (ROOT / p).is_dir():
            f.write(f"{indent}[DIR] {p.name}/\n")
        else:
            try:
                size = (ROOT / p).stat().st_size
                f.write(f"{indent}{p.name}    ({human_size(size)})\n")
            except OSError:
                f.write(f"{indent}{p.name}\n")

    # --------------------------------------------------
    # Largest directories
    # --------------------------------------------------
    f.write("\n\nLARGEST DIRECTORIES\n")
    f.write("=" * 80 + "\n")

    for directory, size in sorted(
        dir_sizes.items(),
        key=lambda x: x[1],
        reverse=True
    )[:100]:
        f.write(f"{human_size(size):>12}    {directory}\n")

    # --------------------------------------------------
    # Largest files
    # --------------------------------------------------
    f.write("\n\nLARGEST FILES\n")
    f.write("=" * 80 + "\n")

    for size, path in sorted(files, reverse=True)[:200]:
        f.write(f"{human_size(size):>12}    {path}\n")

    # --------------------------------------------------
    # File types
    # --------------------------------------------------
    f.write("\n\nSIZE BY FILE TYPE\n")
    f.write("=" * 80 + "\n")

    extensions = sorted(
        extension_sizes,
        key=extension_sizes.get,
        reverse=True,
    )

    for extension in extensions:
        f.write(
            f"{human_size(extension_sizes[extension]):>12}    "
            f"{extension:<15} "
            f"{extension_counts[extension]} files\n"
        )

print("Created repository_inventory.txt")