import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset():
    # Paths
    base_dir = r'd:/Quantum Federated Learning/CT_Scan'
    overview_path = os.path.join(base_dir, 'overview.csv')
    images_dir = os.path.join(base_dir, 'tiff_images')
    
    # Output directories
    output_dirs = {
        'train': os.path.join(base_dir, 'train'),
        'test': os.path.join(base_dir, 'test'),
        'valid': os.path.join(base_dir, 'valid')
    }
    
    # Create directories
    for split in output_dirs:
        for label in ['Pos', 'Neg']:
            os.makedirs(os.path.join(output_dirs[split], label), exist_ok=True)
            
    # Read CSV
    df = pd.read_csv(overview_path)
    
    # Filter valid entries (ensure file exists)
    # Note: overview.csv has 'tiff_name'
    
    valid_data = []
    for index, row in df.iterrows():
        tiff_name = row['tiff_name']
        if pd.isna(tiff_name):
            continue
            
        src_path = os.path.join(images_dir, tiff_name)
        if os.path.exists(src_path):
            label = 'Pos' if row['Contrast'] is True else 'Neg'
            valid_data.append({'tiff_name': tiff_name, 'label': label, 'src_path': src_path})
        else:
            print(f"Warning: File not found {src_path}")

    df_valid = pd.DataFrame(valid_data)
    
    print(f"Total valid images found: {len(df_valid)}")
    
    # Split data
    # Stratify by label to ensure balanced split if possible
    train_df, temp_df = train_test_split(df_valid, test_size=0.2, stratify=df_valid['label'], random_state=42)
    test_df, valid_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    print(f"Train set: {len(train_df)}")
    print(f"Test set: {len(test_df)}")
    print(f"Valid set: {len(valid_df)}")
    
    # Copy files
    def copy_files(dataframe, split_name):
        print(f"Copying files for {split_name}...")
        for _, row in dataframe.iterrows():
            dest_dir = os.path.join(output_dirs[split_name], row['label'])
            dest_path = os.path.join(dest_dir, row['tiff_name'])
            shutil.copy2(row['src_path'], dest_path)
            
    copy_files(train_df, 'train')
    copy_files(test_df, 'test')
    copy_files(valid_df, 'valid')
    
    print("Data split completed successfully.")

if __name__ == "__main__":
    split_dataset()
