import pandas as pd
import os

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

remove_features('./data/TrainingData', './data/TrainingDataV3')