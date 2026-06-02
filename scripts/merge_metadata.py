import pandas as pd

pose_df = pd.read_csv("../Data/filtered_arabesque_poses_master.csv")
meta_df = pd.read_csv("../Data/final_skeleton_dataset.csv")

meta_df['match_video'] = meta_df['filename'].str.split('/').str[-1]

merged_df = pd.merge(
    left=pose_df, 
    right=meta_df, 
    left_on='video',        
    right_on='match_video', 
    how='left'
)

merged_df = merged_df.drop(columns=['match_video'])

missing_meta = merged_df['filename'].isna().sum()
if missing_meta > 0:
    print(f"⚠️ Warning: Miss {missing_meta} rows!")
    print(merged_df[merged_df['filename'].isna()]['video'].unique()[:5])
else:
    print("✅ All pose rows matched!")

merged_df.to_csv("Data/arabesque_analysis_ready.csv", index=False)