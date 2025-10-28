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

            # Specify the reference column
            reference_column = 'right_pos_rel_z'

            # Find the index of the reference column
            ref_col_index = df_raw.columns.get_loc(reference_column)

            # Get the names of columns to drop (all columns after the reference column)
            columns_to_drop = df_raw.columns[ref_col_index + 1:].tolist()

            # Drop the columns
            df_dropped = df_raw.drop(columns=columns_to_drop)

            df_dropped['gait_type'] = df_dropped['gait_type'].str.strip()

            df_dropped.to_csv(processed_full_path, index=False)

remove_features('./data/TrainingData', './data/TrainingDataV2')