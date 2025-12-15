#!/usr/bin/env python3
"""
Fix relative imports to absolute imports in src/cad package.
Changes:
  from ..domain.X import Y  → from cad.domain.X import Y
  from ...X import Y        → from cad.X import Y (for deeply nested files)
"""
import re
from pathlib import Path

def fix_imports_in_file(file_path: Path):
    """Fix relative imports in a single file."""
    content = file_path.read_text()
    original = content

    # Get the relative path from src/cad/
    relative_to_cad = file_path.relative_to(Path("src/cad"))
    depth = len(relative_to_cad.parts) - 1  # Subtract 1 for the file itself

    # Pattern 1: from ..X import Y (one level up)
    # Pattern 2: from ...X import Y (two levels up)
    # etc.

    # Replace from most dots to least to avoid conflicts
    for num_dots in range(5, 0, -1):
        dots = "." * num_dots
        pattern = rf"from {re.escape(dots)}(\w+)"

        def replacer(match):
            module = match.group(1)
            # Calculate how many levels up we need to go
            levels_up = num_dots - 1
            if levels_up <= depth:
                return f"from cad.{module}"
            else:
                # Going up too many levels - keep as is
                return match.group(0)

        content = re.sub(pattern, replacer, content)

    if content != original:
        file_path.write_text(content)
        return True
    return False

def main():
    src_cad = Path("src/cad")
    if not src_cad.exists():
        print("❌ src/cad directory not found!")
        return

    # Find all Python files with relative imports
    py_files = list(src_cad.rglob("*.py"))
    fixed_count = 0

    for py_file in py_files:
        content = py_file.read_text()
        if "from .." in content:
            print(f"📝 Fixing {py_file.relative_to('src/cad')}")
            if fix_imports_in_file(py_file):
                fixed_count += 1
                print(f"   ✅ Fixed")

    print(f"\n🎉 Fixed {fixed_count} files!")

if __name__ == "__main__":
    main()
