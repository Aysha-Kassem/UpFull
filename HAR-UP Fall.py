import pandas as pd
import os

# --- Configuration ---

# Root path where the Subject folders (S01, S02, etc.) are located
ROOT_PATH = 'upfall_data'  

# Define Activity IDs and their corresponding labels
# Fall Activities (Label = 1)
FALL_ACTIVITIES = ['A07', 'A08', 'A09', 'A10', 'A11']

# Activities of Daily Living (ADL) (Label = 0)
# A01: Walking, A02: Standing, A03: Picking up object, A04: Sitting, 
# A05: Lying, A06: Jumping
ADL_ACTIVITIES = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06'] 

# Create a consolidated list for quick label lookup
ACTIVITY_LABELS = {activity: 1 for activity in FALL_ACTIVITIES}
ACTIVITY_LABELS.update({activity: 0 for activity in ADL_ACTIVITIES})

# ----------------------------------------------------
# Core Function to Load and Label Data
# ----------------------------------------------------

def load_upfall_dataset(root_dir, activity_labels_map):
    """
    Reads all CSV files from the structured UP-Fall dataset, labels them, 
    and combines them into a single pandas DataFrame.
    
    Args:
        root_dir (str): Path to the main data folder (e.g., 'upfall_data').
        activity_labels_map (dict): Map of Activity ID (A01, A07) to Label (0 or 1).
        
    Returns:
        pd.DataFrame: A single DataFrame with all data and classification labels.
    """
    all_data = []
    
    # 1. Traverse Subject folders
    for subject_folder in os.listdir(root_dir):
        if not subject_folder.startswith('S') or not os.path.isdir(os.path.join(root_dir, subject_folder)):
            continue
        
        subject_path = os.path.join(root_dir, subject_folder)

        # 2. Traverse Activity folders
        for activity_folder in os.listdir(subject_path):
            activity_path = os.path.join(subject_path, activity_folder)
            
            # Check if the folder is a known activity and get the label
            if activity_folder not in activity_labels_map:
                continue 
            
            label = activity_labels_map[activity_folder]
            
            # 3. Traverse Trial files (CSVs)
            for trial_file in os.listdir(activity_path):
                if trial_file.endswith('.csv'):
                    file_path = os.path.join(activity_path, trial_file)
                    print(f"Reading: {file_path}")

                    try:
                        # Read CSV file
                        df = pd.read_csv(file_path)
                        
                        # Add metadata and classification label
                        df['Subject'] = subject_folder
                        df['Activity_ID'] = activity_folder
                        df['Label'] = label  # 1 for Fall, 0 for ADL
                        
                        all_data.append(df)
                        
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
                        
    # Combine all parts into one DataFrame
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df
    else:
        print("No data files found. Check your ROOT_PATH and folder structure.")
        return pd.DataFrame()


# ----------------------------------------------------
# Program Execution
# ----------------------------------------------------
if __name__ == "__main__":
    
    # 1. Load the entire dataset
    full_dataset_df = load_upfall_dataset(ROOT_PATH, ACTIVITY_LABELS)
    
    # 2. Summarize the data
    if not full_dataset_df.empty:
        print("\n" + "="*50)
        print("Dataset Loading Complete.")
        print("First 5 rows of the combined data:")
        print(full_dataset_df.head())
        
        print(f"\nTotal rows loaded: {len(full_dataset_df)}")
        
        # Display the count of Fall (1) vs. ADL (0) activities
        print("\nLabel Distribution (0=ADL, 1=Fall):")
        print(full_dataset_df['Label'].value_counts())
        
        # Optional: Save the merged DataFrame to a single file
        output_file = 'UPFALL_Full_Merged_Data.csv'
        full_dataset_df.to_csv(output_file, index=False)
        print(f"\nMerged data saved to: {output_file}")
    
    else:
        print("Exiting program.")