import pandas as pd
from pathlib import Path

BASE_DIR = Path("results")
OUTPUT_CSV_PATH = "Data/filtered_arabesque_poses_master.csv"

all_filtered_data = []

for edition_dir in BASE_DIR.glob("edition*"):
    if not edition_dir.is_dir():
        continue
        
    frames_dir = edition_dir / "frames"
    if not frames_dir.exists():
        continue
        
 
    valid_keys_mapping = {}
    for f in frames_dir.glob("*"):
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            match_key = f.name.split('_score_')[0] 
            valid_keys_mapping[match_key] = f.name
            
    if not valid_keys_mapping:
        continue

    for csv_file in edition_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            
            if not {'video', 'peak_id', 'frame_idx'}.issubset(df.columns):
                print(f"⚠️ {csv_file.name}: No video, peak_id or frame_idx column, skip.")
                continue
            
            df['match_key'] = (
                df['video'].astype(str) + 
                "_peak_" + df['peak_id'].astype(str) + 
                "_frame_" + df['frame_idx'].astype(str)
            )
        
            filtered_df = df[df['match_key'].isin(valid_keys_mapping.keys())].copy()
            
            if not filtered_df.empty:
                filtered_df['valid_img_name'] = filtered_df['match_key'].map(valid_keys_mapping)
                
                filtered_df['source_edition'] = edition_dir.name
                
                filtered_df = filtered_df.drop(columns=['match_key'])
                
                all_filtered_data.append(filtered_df)
                
        except Exception as e:
            print(f"❌ {csv_file}: {e}")

if all_filtered_data:
    final_df = pd.concat(all_filtered_data, ignore_index=True)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Success! Extract {len(final_df)} valid rows.")
    print(f"📁 Write to: {OUTPUT_CSV_PATH}")
else:
    print("\n⚠️ No matching data!")