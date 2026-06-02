import os
import glob
import pandas as pd

def cleanup_videos(dataset_csv, start_edition, end_edition, base_video_dir):
    """
    Deletes .mp4 files in edition folders that are NOT listed in the dataset CSV.
    """
    if not os.path.exists(dataset_csv):
        print(f"[ERROR] Dataset CSV not found at: {dataset_csv}")
        return

    print(f"[INFO] Loading dataset from {dataset_csv}...")
    df = pd.read_csv(dataset_csv)
    # The 'filename' column contains paths like 'edition19/edition19_sequence_13.mp4'
    allowed_filenames = set(df['filename'].tolist())

    print(f"[INFO] Starting cleanup for Editions {start_edition} to {end_edition}...")
    
    total_deleted = 0
    
    for i in range(start_edition, end_edition + 1):
        edition_folder_name = f"edition{i}"
        edition_path = os.path.join(base_video_dir, edition_folder_name)
        
        if not os.path.exists(edition_path):
            # Check if zip exists (just for logging)
            if os.path.exists(f"{edition_path}.zip"):
                print(f"[SKIP] {edition_folder_name}: Folder not found, but ZIP exists.")
            continue

        print(f"[INFO] Checking {edition_folder_name}...")
        
        # Get all mp4 files in this edition folder
        video_files = glob.glob(os.path.join(edition_path, "*.mp4"))
        
        edition_deleted = 0
        for file_path in video_files:
            vid_name = os.path.basename(file_path)
            # Reconstruct the relative path as it appears in the CSV
            rel_path = f"{edition_folder_name}/{vid_name}"
            
            if rel_path not in allowed_filenames:
                try:
                    os.remove(file_path)
                    print(f"   [DELETE] {rel_path}")
                    edition_deleted += 1
                except Exception as e:
                    print(f"   [ERROR] Error deleting {rel_path}: {e}")
        
        if edition_deleted > 0:
            print(f"   [DONE] Deleted {edition_deleted} unlisted videos in {edition_folder_name}.")
        else:
            print(f"   [CLEAN] {edition_folder_name}: All files match CSV or folder already clean.")
            
        total_deleted += edition_deleted

    print("==================================================")
    print(f"[FINISH] Cleanup complete. Total videos deleted: {total_deleted}")
    print("==================================================")

if __name__ == "__main__":
    # CONFIGURATION
    DATASET_PATH = r"Data\final_skeleton_dataset.csv"
    START_EDITION = 19
    END_EDITION = 30
    BASE_VIDEO_DIR = "videos"
    
    cleanup_videos(DATASET_PATH, START_EDITION, END_EDITION, BASE_VIDEO_DIR)
