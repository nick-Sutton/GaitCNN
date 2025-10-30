import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import glob
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

from dataset.gait_dataset import GaitDataset

class GaitPreprocessor:
    """
    Preprocessor for gait sensor data from CSV files.
    Handles normalization and windowing for the GaitCNN model.
    """
    
    def __init__(self, window_length=100, stride=50, normalize=True):
        """
        Args:
            window_length: Length of time window (default 100 to match model)
            stride: Step size for sliding window (50 = 50% overlap)
            normalize: Whether to normalize data to [0, 1]
        """
        self.window_length = window_length
        self.stride = stride
        self.normalize = normalize
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.label_encoder = LabelEncoder()
        self.support_type_encoder = LabelEncoder()
        self.is_fitted = False
        self.labels_fitted = False
        
    def load_csv_files(self, file_paths):
        """
        Load multiple CSV files and concatenate them.
        
        Args:
            file_paths: List of CSV file paths or glob pattern (e.g., 'data/*.csv')
        
        Returns:
            DataFrame with all data
        """
        if isinstance(file_paths, str):
            file_paths = glob.glob(file_paths)
        
        dfs = []
        for path in file_paths:
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"Loaded {path}: {len(df)} rows, {len(df.columns)} columns")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal combined: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        return combined_df
    
    def prepare_data(self, df, label_column='gait_type'):
        """
        Prepare data: separate features and labels, normalize if needed.
        
        Args:
            df: DataFrame with sensor data
            label_column: Name of the column containing class labels
        
        Returns:
            features (numpy array), labels (numpy array)
        """
        # Separate features and labels
        if label_column in df.columns:
            labels = df[label_column].values
            features = df.drop(columns=[label_column]).values
        else:
            raise ValueError(f"Label column '{label_column}' not found in DataFrame")
        
        print(f"Features shape: {features.shape}")
        print(f"Number of channels (sensors): {features.shape[1]}")

        # Encode string labels to integers
        if not self.labels_fitted:
            labels = self.label_encoder.fit_transform(labels)
            self.labels_fitted = True
            print(f"Fitted label encoder on classes: {self.label_encoder.classes_}")
        else:
            labels = self.label_encoder.transform(labels)
        
        print(f"Unique classes: {np.unique(labels)} -> {self.label_encoder.classes_}")

        # Encode support type BEFORE dropping columns
        support_type_values = None
        if 'support_type' in df.columns:
            if not hasattr(self, 'support_type_fitted'):
                support_type_values = self.support_type_encoder.fit_transform(df['support_type'])
                self.support_type_fitted = True
                print(f"Fitted support_type encoder on classes: {self.support_type_encoder.classes_}")
            else:
                support_type_values = self.support_type_encoder.transform(df['support_type'])
            
            print(f"Support Types: {np.unique(support_type_values)} -> {self.support_type_encoder.classes_}")
            
            # Remove support_type temporarily for normalization
            df = df.drop(columns=['support_type'])
        
        # Extract features (without support_type and label)
        features = df.drop(columns=[label_column]).values

        # Normalize features to [0, 1]
        if self.normalize:
            if not self.is_fitted:
                features = self.scaler.fit_transform(features)
                self.is_fitted = True
                print("Fitted scaler on data")
            else:
                features = self.scaler.transform(features)
                print("Applied existing scaler")
            
            print(f"Normalized - Min: {features.min():.4f}, Max: {features.max():.4f}")

        # Add support_type back as a separate column (not normalized)
        if support_type_values is not None:
            features = np.column_stack([features, support_type_values])
            print(f"Added support_type column (not normalized)")
        
        print(f"Final features shape: {features.shape}")
        print(f"Number of channels (sensors): {features.shape[1]}")
        
        return features, labels
    
    def create_windows(self, features, labels):
        """
        Create sliding time windows from sequential data.
        
        Args:
            features: Array of shape (timesteps, channels)
            labels: Array of shape (timesteps,)
        
        Returns:
            windowed_features: Array of shape (n_windows, window_length, channels)
            windowed_labels: Array of shape (n_windows,) - label for each window
        """
        n_timesteps, n_channels = features.shape
        
        # Calculate number of windows
        n_windows = (n_timesteps - self.window_length) // self.stride + 1
        
        windowed_features = []
        windowed_labels = []
        
        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_length
            
            # Extract window
            window = features[start_idx:end_idx, :]
            
            window_labels = labels[start_idx:end_idx]

            # Using majority vote:
            label = np.bincount(window_labels.astype(int)).argmax()
            
            windowed_features.append(window)
            windowed_labels.append(label)
        
        windowed_features = np.array(windowed_features)
        windowed_labels = np.array(windowed_labels)
        
        print(f"\nCreated {n_windows} windows:")
        print(f"  Window shape: ({self.window_length}, {n_channels})")
        print(f"  Output shape: {windowed_features.shape}")
        print(f"  Labels shape: {windowed_labels.shape}")
        
        return windowed_features, windowed_labels
    
def prepare_dataloaders(data_files, 
                       window_length=100, 
                       stride=50,
                       batch_size=32, 
                       label_column='gait_type',
                       test_size=0.2,
                       random_state=42,
                       stratify=True):
    """
    Complete pipeline: load CSVs -> normalize -> window -> split -> create DataLoaders
    
    Args:
        data_files: CSV file paths (list or glob pattern)
        window_length: Time window length (should match model input_length)
        stride: Sliding window stride
        batch_size: Batch size for DataLoader
        label_column: Name of label column in CSV
        test_size: Fraction of data to use for testing (0.2 = 20%)
        random_state: Random seed for reproducibility
        stratify: If True, maintains class distribution in train/test split
    
    Returns:
        train_loader, test_loader, preprocessor
    """
    preprocessor = GaitPreprocessor(window_length=window_length, stride=stride)
    
    # Load all data
    print("=" * 50)
    print("LOADING DATA")
    print("=" * 50)
    df = preprocessor.load_csv_files(data_files)
    features, labels = preprocessor.prepare_data(df, label_column)
    windows, window_labels = preprocessor.create_windows(features, labels)
    
    # Split into train and test
    print("\n" + "=" * 50)
    print("SPLITTING DATA")
    print("=" * 50)
    
    stratify_labels = window_labels if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        windows, 
        window_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels
    )
    
    print(f"Train set: {len(X_train)} windows")
    print(f"Test set: {len(X_test)} windows")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")

    # Get class names from label encoder
    class_names = preprocessor.label_encoder.classes_.tolist()
    print(f"Class names: {class_names}")
    
    # Create datasets and dataloaders
    train_dataset = GaitDataset(X_train, y_train)
    test_dataset = GaitDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, test_loader, preprocessor