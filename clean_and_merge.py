import os
import shutil
import hashlib
from collections import defaultdict
import random

def get_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def clean_and_merge_datasets(src_dirs, dest_dir, split_ratio=0.8):
    """
    Cleans duplicate images and merges multiple datasets into one unified folder.
    """
    # Dictionary to store unique files per class: class_name -> set of file hashes
    unique_hashes = defaultdict(set)
    # Dictionary to store list of filepaths per class: class_name -> list of paths
    class_files = defaultdict(list)

    print("Scanning datasets...")
    total_scanned = 0
    total_duplicates = 0

    for src_dir in src_dirs:
        if not os.path.exists(src_dir):
            print(f"Warning: Directory {src_dir} does not exist. Skipping.")
            continue
            
        for class_name in os.listdir(src_dir):
            class_path = os.path.join(src_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for filename in os.listdir(class_path):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                filepath = os.path.join(class_path, filename)
                total_scanned += 1
                
                file_hash = get_file_hash(filepath)
                if file_hash in unique_hashes[class_name]:
                    total_duplicates += 1
                else:
                    unique_hashes[class_name].add(file_hash)
                    class_files[class_name].append(filepath)

    print(f"\nScan complete.")
    print(f"Total files scanned: {total_scanned}")
    print(f"Identical duplicates found and removed: {total_duplicates}")
    print(f"Total unique files to merge: {total_scanned - total_duplicates}")

    # Prepare destination directories
    if os.path.exists(dest_dir):
        print(f"Removing existing output directory: {dest_dir}")
        shutil.rmtree(dest_dir)
        
    os.makedirs(dest_dir, exist_ok=True)

    print(f"\nCopying files into a single directory...")
    
    for class_name, filepaths in class_files.items():
        os.makedirs(os.path.join(dest_dir, class_name), exist_ok=True)
        print(f"Class '{class_name}': {len(filepaths)} files")
        
        # Copy all files
        for i, filepath in enumerate(filepaths):
            ext = os.path.splitext(filepath)[1]
            new_filename = f"{class_name}_{i+1:05d}{ext}"
            shutil.copy2(filepath, os.path.join(dest_dir, class_name, new_filename))

    print(f"\nData cleaning and merging completed successfully. Saved to {dest_dir}")

if __name__ == "__main__":
    src_directories = [
        os.path.join('data', 'ADNI'),
        os.path.join('data', 'OASIS3'),
        os.path.join('data', 'AIBL')
    ]
    destination_directory = os.path.join('data', 'all_data')
    
    # Set fixed seed for consistent train/val split over multiple runs
    random.seed(42)
    
    clean_and_merge_datasets(src_directories, destination_directory)
