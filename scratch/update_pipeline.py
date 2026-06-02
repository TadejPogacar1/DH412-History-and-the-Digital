import json
import os

notebook_path = r'd:\DH412-History-and-the-Digital\pipeline_arabesque.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update imports (first cell)
nb['cells'][0]['source'] = [
    "import os\n",
    "import glob\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from mpl_toolkits.mplot3d import Axes3D\n",
    "import cv2\n",
    "import librosa\n",
    "from scipy.signal import find_peaks, savgol_filter\n",
    "from scipy.ndimage import median_filter\n",
    "from scipy.ndimage import gaussian_filter1d\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.cluster import AgglomerativeClustering\n",
    "from sklearn.metrics import silhouette_score\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from moviepy.editor import VideoFileClip\n",
    "from mmpose.apis import MMPoseInferencer\n",
    "import json\n",
    "import csv\n",
    "import pickle\n",
    "import shutil\n"
]

# 2. Find and update process_video_folder, and add cleanup_video_folder
# It's in the same large cell as the classes
main_cell = nb['cells'][0]
source = main_cell['source']

# Find where process_video_folder starts
start_idx = -1
for i, line in enumerate(source):
    if "def process_video_folder(input_folder, output_folder):" in line:
        start_idx = i
        break

if start_idx != -1:
    # We'll replace from start_idx to the end of the cell if it's the end, 
    # but wait, the cell continues. 
    # Actually, the cell ends with print("\\n✅ Batch Processing Complete.")
    end_idx = -1
    for i in range(start_idx, len(source)):
        if 'print("\\n✅ Batch Processing Complete.")' in source[i]:
            end_idx = i + 1
            break
    
    if end_idx != -1:
        new_source = source[:start_idx-1] # Remove the comment too
        
        cleanup_code = [
            "\n",
            "# ==========================================\n",
            "# Main Pipeline\n",
            "# ==========================================\n",
            "def cleanup_video_folder(folder_path, allowed_filenames):\n",
            "    \"\"\"\n",
            "    Deletes .mp4 files in folder_path that are NOT in the allowed_filenames list.\n",
            "    \"\"\"\n",
            "    print(f\"🧹 Cleaning up folder: {folder_path}...\")\n",
            "    if not os.path.exists(folder_path):\n",
            "        print(f\"   ⚠️ Folder {folder_path} does not exist. Skipping cleanup.\")\n",
            "        return\n",
            "\n",
            "    all_files = glob.glob(os.path.join(folder_path, \"*.mp4\"))\n",
            "    deleted_count = 0\n",
            "    \n",
            "    for file_path in all_files:\n",
            "        vid_name = os.path.basename(file_path)\n",
            "        # Construct relative path to match CSV (e.g., 'edition19/video.mp4')\n",
            "        rel_path = f\"{os.path.basename(folder_path)}/{vid_name}\"\n",
            "        \n",
            "        if rel_path not in allowed_filenames:\n",
            "            try:\n",
            "                os.remove(file_path)\n",
            "                deleted_count += 1\n",
            "            except Exception as e:\n",
            "                print(f\"   ❌ Error deleting {vid_name}: {e}\")\n",
            "    \n",
            "    print(f\"   ✅ Cleanup complete. Deleted {deleted_count} unlisted videos.\")\n",
            "\n"
        ]
        
        process_code = [
            "def process_video_folder(input_folder, output_folder, allowed_filenames):\n",
            "    print(\"==================================================\")\n",
            "    print(f\"🚀 PROCESSING FOLDER: {input_folder}\")\n",
            "    print(\"==================================================\")\n",
            "    \n",
            "    video_files = glob.glob(os.path.join(input_folder, \"*.mp4\"))\n",
            "    \n",
            "    # Final filter to double check we only process what's allowed\n",
            "    video_files = [f for f in video_files if f\"{os.path.basename(input_folder)}/{os.path.basename(f)}\" in allowed_filenames]\n",
            "    \n",
            "    if not video_files:\n",
            "        print(f\"❌ No matching videos found in {input_folder} after filtering!\")\n",
            "        return\n",
            "        \n",
            "    print(f\"📁 Found {len(video_files)} target videos for processing.\")\n"
        ]
        
        # Add the rest of process_video_folder logic (which starts after the print in the original)
        rest_of_process = source[start_idx+10:end_idx] # Adjust index based on where the original logic continues
        # Actually, let's just keep the original logic and only change the signature and start
        
        # Re-fetch the rest of the function correctly
        # The original code had 10 lines of prints/setup before the loop
        # Let's just find the loop
        loop_start = -1
        for i in range(start_idx, end_idx):
            if "for idx, video_path in enumerate(video_files):" in source[i]:
                loop_start = i
                break
        
        if loop_start != -1:
            main_cell['source'] = source[:start_idx-1] + cleanup_code + process_code + source[loop_start-4:end_idx] # Keep some setup lines
        else:
            print("Could not find loop start")

# 3. Update the execution cell
# Find cell with process_video_folder("./videos/edition19", "./results/edition19")
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('process_video_folder("./videos/edition19"' in line for line in cell['source']):
        cell['source'] = [
            "# ==========================================\n",
            "# RUN BATCH PROCESSING FOR EDITIONS 19-30\n",
            "# ==========================================\n",
            "dataset_path = \"Data/final_skeleton_dataset.csv\"\n",
            "if os.path.exists(dataset_path):\n",
            "    df_dataset = pd.read_csv(dataset_path)\n",
            "    # Extract filenames as a set for O(1) lookup\n",
            "    allowed_filenames = set(df_dataset['filename'].tolist())\n",
            "    \n",
            "    # Define range\n",
            "    start_edition = 19\n",
            "    end_edition = 30\n",
            "    \n",
            "    for i in range(start_edition, end_edition + 1):\n",
            "        edition_folder = f\"./videos/edition{i}\"\n",
            "        output_folder = f\"./results/edition{i}\"\n",
            "        \n",
            "        if os.path.exists(edition_folder):\n",
            "            # Step 1: Cleanup (Delete videos not in CSV)\n",
            "            cleanup_video_folder(edition_folder, allowed_filenames)\n",
            "            \n",
            "            # Step 2: Process\n",
            "            os.makedirs(output_folder, exist_ok=True)\n",
            "            process_video_folder(edition_folder, output_folder, allowed_filenames)\n",
            "        else:\n",
            "            # Check if zip exists\n",
            "            if os.path.exists(f\"{edition_folder}.zip\"):\n",
            "                print(f\"📦 Edition {i} found as ZIP. Please unzip it to {edition_folder} to process.\")\n",
            "else:\n",
            "    print(f\"❌ Dataset file not found at {dataset_path}!\")\n"
        ]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
