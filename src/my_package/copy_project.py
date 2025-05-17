import shutil
import os

def copy_subdirs(root_directory, destination_dir):
    for subdir in ['results', 'data']:
        src = os.path.join(root_directory, subdir)
        dst = os.path.join(destination_dir, subdir)
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"Copied '{src}' to '{dst}'")
        else:
            print(f"Warning: '{subdir}' not found in '{root_directory}'")
