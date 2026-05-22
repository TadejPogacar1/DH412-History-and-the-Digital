import pandas as pd
import numpy as np
import json
import os
import re

def clean_json_path(path):
    if pd.isna(path):
        return path
        
    path = str(path)
    match = re.search(r'(results[\\/].*)', path)
    if match:
        return match.group(1).replace('\\', '/')
    return path.replace('\\', '/')

def calculate_new_angles(row):
    json_path = row['clean_json_path']
    raised_leg = str(row['raised_leg']).strip().lower() 
    
    try:
        if not os.path.exists(json_path):
            return pd.Series({'thoracic_neck_angle': np.nan, 'support_torso_angle': np.nan})
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        skel = data['skeleton_3d']
        
        pelvis = np.array(skel['Pelvis'])
        neck = np.array(skel['Neck'])
        thorax = np.array(skel['Thorax'])
        head = np.array(skel['Head'])
        l_ankle = np.array(skel['L_Ankle'])
        r_ankle = np.array(skel['R_Ankle'])
        l_hip = np.array(skel['L_Hip'])
        r_hip = np.array(skel['R_Hip'])
        
        # Thoracic-Neck Line
        thorax_to_neck = neck - thorax
        neck_to_head = head - neck
        
        cosine_tn = np.dot(thorax_to_neck, neck_to_head) / (np.linalg.norm(thorax_to_neck) * np.linalg.norm(neck_to_head))
        cosine_tn = np.clip(cosine_tn, -1.0, 1.0) 
        tn_angle = np.degrees(np.arccos(cosine_tn))
        
        # Support-torso angle
        if raised_leg in ['left', 'l']:
            support_ankle = r_ankle   
            support_hip = r_hip
        elif raised_leg in ['right', 'r']:
            support_ankle = l_ankle   
            support_hip = l_hip
        else:
            support_ankle = l_ankle if l_ankle[2] < r_ankle[2] else r_ankle
        
        vec_torso = neck - pelvis
        vec_support_leg = support_ankle - support_hip
        
        cosine_hip = np.dot(vec_torso, vec_support_leg) / (np.linalg.norm(vec_torso) * np.linalg.norm(vec_support_leg))
        cosine_hip = np.clip(cosine_hip, -1.0, 1.0)
        hip_angle = np.degrees(np.arccos(cosine_hip))
        
        return pd.Series({'thoracic_neck_angle': tn_angle, 'support_torso_angle': hip_angle})

    except Exception as e:
        return pd.Series({'thoracic_neck_angle': np.nan, 'support_torso_angle': np.nan})

if __name__ == "__main__":
    csv_input_path = 'Data/arabesque_analysis_ready.csv'
    csv_output_path = 'Data/arabesque_analysis_ready_updated.csv'
    
    print(f"Loading: {csv_input_path}")
    df = pd.read_csv(csv_input_path)

    old_columns_to_remove = ['thoracic_neck_line']
    existing_cols_to_drop = [col for col in old_columns_to_remove if col in df.columns]
    if existing_cols_to_drop:
        print(f"Remove: {existing_cols_to_drop}")
        df.drop(columns=existing_cols_to_drop, inplace=True)
    
    df['clean_json_path'] = df['json_path'].apply(clean_json_path)
    
    new_features = df.apply(calculate_new_angles, axis=1)
    
    df = pd.concat([df, new_features], axis=1)
    
    df['json_path'] = df['clean_json_path']
    df.drop(columns=['clean_json_path'], inplace=True)
    
    print(f"Save to: {csv_output_path}")
    df.to_csv(csv_output_path, index=False)