import os
import zipfile

zip_filename = "7115056142_hw2.zip"

files_to_include = [
    "7115056142_hw2.py",
    "7115056142_hw2.ipynb",
    "7115056142_hw2_report.pdf",
    "7115056142_hw2_report.md",
    "GPT_dialogue.pdf",
    "README.md",
    "app.py",
    "generate_data.py"
]

folders_to_include = [
    "data",
    "figures",
    "web"
]

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for f in files_to_include:
        if os.path.exists(f):
            zipf.write(f, arcname=f"7115056142_hw2/{f}")
            print(f"Added file: {f}")
            
    for folder in folders_to_include:
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, ".")
                    zipf.write(file_path, arcname=f"7115056142_hw2/{rel_path}")
                    print(f"Added: {rel_path}")

print(f"\nCreated {zip_filename} successfully! Size: {os.path.getsize(zip_filename):,} bytes")
