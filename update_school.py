import os

old_school = "國立成功大學"
new_school = "國立中興大學"

files = [
    "web/index.html",
    "docs/index.html",
    "7115056142_hw2_report.md",
    "7115056142_hw2.py",
    "build_notebook.py",
    "generate_pdf_report.py"
]

for fname in files:
    if os.path.exists(fname):
        with open(fname, 'r', encoding='utf-8') as f:
            content = f.read()
        content = content.replace(old_school, new_school)
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {fname} to {new_school}")

print("All occurrences updated!")
