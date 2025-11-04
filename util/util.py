import pandas as pd
import os

def reformat_motive_csv(input_file, output_file):
    raw = pd.read_csv(input_file, header=None)

    # Detect header row
    header_row_idx = raw.index[raw.iloc[:, 0] == "Frame"][0]

    # Load data without motive header
    data = pd.read_csv(input_file, skiprows=header_row_idx)

    # Frame/Time
    frame_time = data.iloc[:, :2].copy()
    frame_time.columns = ["Frame", "Time (Seconds)"]

    # Rigid body names
    name_row_idx = raw.index[raw.iloc[:, 1] == "Name"][0]
    rigid_body_names = raw.iloc[name_row_idx, 2:].tolist()

    # Rotation/Position labels
    label_row_idx = raw.index[raw.iloc[:, 1] == "ID"][0] + 2
    labels_row = raw.iloc[label_row_idx, 2:].tolist()

    # Axis row
    axes_row = raw.iloc[header_row_idx, 2:].tolist()

    # Build column RigidBodyName:Rotation/Position:Axis
    col_names = []
    for rb_name, label, axis in zip(rigid_body_names, labels_row, axes_row):
        if pd.isna(rb_name):  # skip blanks
            continue
        col_names.append(f"{rb_name}:{label}:{axis}")

    # Assign columns
    rigid_body_data = data.iloc[:, 2:]
    rigid_body_data.columns = col_names

    rigid_body_data = rigid_body_data.drop(index=0).reset_index(drop=True)
    frame_time = frame_time.drop(index=0).reset_index(drop=True)

    # Reformatted DataFrame
    final_df = pd.concat([frame_time, rigid_body_data], axis=1)

    # Save
    final_df.to_csv(output_file, index=False)
    print(f"Reformatted CSV saved to {output_file}")


def generate_reformatted_data(raw_files_dir, processed_files_dir):

    # Create the output directory if it does not already exist
    os.makedirs(processed_files_dir, exist_ok=True)

    # Loop through input directory and run the reformatter on each file
    for file_name in os.listdir(raw_files_dir):
        raw_full_path = os.path.join(raw_files_dir, file_name)
        processed_full_path = os.path.join(processed_files_dir, file_name)
        if os.path.isfile(raw_full_path):
            reformat_motive_csv(raw_full_path, processed_full_path)


# Human Gaits 
# Stand
# Quasi steady
# trot 
# running
def lable_data(input_file, output_file):
    gait_keywords = {
        'quasi': 'quasi',
        'walk': 'walk',
        'jog': 'jog',
        'stand': 'stand',
        'stepup': 'step_acsent',
        'stepdown': 'step_decsent',
        'stairup': 'stair_acsent',
        'stairdown': 'stair_decsent'
    }

    df = pd.read_csv(input_file)

    filename = os.path.basename(input_file)
    file_class = filename.split('_')[0].lower()

    # Map to standard gait label
    gait_type = gait_keywords.get(file_class, file_class)
    
    # Add label to all rows
    df['gait_type'] = gait_type

    # Save
    df.to_csv(output_file, index=False)
    print(f"Labled CSV saved to {output_file}")
    

def generate_lable_data(raw_files_dir, processed_files_dir):

    # Create the output directory if it does not already exist
    os.makedirs(processed_files_dir, exist_ok=True)

    for file_name in os.listdir(raw_files_dir):
        raw_full_path = os.path.join(raw_files_dir, file_name)
        processed_full_path = os.path.join(processed_files_dir, file_name)
        if os.path.isfile(raw_full_path):
            lable_data(raw_full_path, processed_full_path)

def extract_gait_label_from_filename(filename):
    """
    Extract gait label from filename.
    Example: 'Walk_forward_001.csv' -> 'walk'
    """
    gait_keywords = {
        'quasi': 'quasi',
        'walk': 'walk',
        'jog': 'jog',
        'stand': 'stand',
        'stepup': 'step_acsent',
        'stepdown': 'step_decsent',
        'stairup': 'step_acsent',
        'stairdown': 'step_decsent'
    }
    
    first_word = os.path.basename(filename).split('_')[0].lower()
    return gait_keywords.get(first_word, first_word)

def remove_features(input_file_dir: str, output_file_dir: str):
    # Create the output directory if it does not already exist
    os.makedirs(output_file_dir, exist_ok=True)

    # Loop through input directory and run the reformatter on each file
    for file_name in os.listdir(input_file_dir):
        raw_full_path = os.path.join(input_file_dir, file_name)
        processed_full_path = os.path.join(output_file_dir, file_name)
        if os.path.isfile(raw_full_path):
            df_raw = pd.read_csv(raw_full_path)

            # Cols to drop
            pos_cols = ['root_position_x','root_position_y','root_position_z',
                        'root_orientation_x','root_orientation_y','root_orientation_z']

            # Drop the columns
            df_dropped = df_raw.drop(columns=pos_cols)

            # remove any spaces in gait_type
            df_dropped['gait_type'] = df_dropped['gait_type'].str.strip()

            df_dropped.to_csv(processed_full_path, index=False)

#remove_features('./data/TrainingData', './data/TrainingDataV3')
#generate_reformatted_data('./data/TrackingDataV5', './data/UnlabledDataV5')
generate_lable_data('./data/UnlabeledDataV5', './data/LabeledDataV5')