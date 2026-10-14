import os
import shutil

old_id = "7114056001"
new_id = "7115056142"

files_to_update = [
    "build_notebook.py",
    "generate_pdf_report.py",
    "make_zip.py",
    "7114056001_hw2_report.md",
    "7114056001_hw2.py",
    "web/index.html",
    "web/app.js",
    "app.py",
    "README.md"
]

for fname in files_to_update:
    if os.path.exists(fname):
        with open(fname, 'r', encoding='utf-8') as f:
            content = f.read()
        content = content.replace(old_id, new_id)
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated content in {fname}")

# Rename files from 7114056001 to 7115056142
renames = [
    ("7114056001_hw2.py", f"{new_id}_hw2.py"),
    ("7114056001_hw2_report.md", f"{new_id}_hw2_report.md"),
]

for src, dst in renames:
    if os.path.exists(src):
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(src, dst)
        print(f"Renamed {src} -> {dst}")

# Remove old generated files so they can be freshly rebuilt
old_files_to_remove = [
    "7114056001_hw2.ipynb",
    "7114056001_hw2_report.pdf",
    "7114056001_hw2.zip"
]
for f in old_files_to_remove:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed old artifact: {f}")

print("\nAll files successfully updated to 7115056142!")
