
# fix_imports.py
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to sys.path: {project_root}")

# Optionally, add the 'src' directory itself if needed, though adding the root is usually sufficient
# src_path = os.path.abspath('./src')
# if src_path not in sys.path:
#    sys.path.insert(0, src_path)
#    print(f"Added src directory to sys.path: {src_path}")

# You can verify the path includes your project root now
# print("\nCurrent sys.path:")
# for p in sys.path:
#     print(p)
# print("-" * 20)
    