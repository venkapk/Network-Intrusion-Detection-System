import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import os
import datetime
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split

def process_dataset(dataset_path, output_dir, prefix, class_type = 'binary', correlation_threshold = 0.1):
    """
    Process a raw network traffic dataset for model training.
    
    Performs data cleaning, feature selection, and outlier removal to prepare
    raw network traffic data for machine learning.
    
    Parameters:
    dataset_path (str): Path to the raw dataset CSV
    output_dir (str): Directory to save processed data and artifacts
    prefix (str): Prefix for saved files (typically dataset name)
    class_type (str): Classification type ('binary', 'multiclass_3', 'multiclass_4', etc.)
    correlation_threshold (float): Minimum correlation with target to keep a feature
    
    Returns:
    pd.DataFrame: Processed dataset with selected features
    """
    start_time = datetime.datetime.now()
    print(f"Processing {prefix} dataset: {dataset_path}")
    
    # Create output directory for preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    
    # Read raw data
    df = pd.read_csv(dataset_path)
    initial_shape = df.shape
    
    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove duplicate network flows
    df = df.drop_duplicates()

    # Drop rows with missing values
    df = df.dropna()

    # Handle infinite values in numeric fields
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].max())

    # Remove constant features that provide no discriminatory value
    constant_columns = [col for col in numeric_columns if df[col].nunique() == 1]
    df = df.drop(columns=constant_columns)

    # Create numeric target variable based on class_type
    if class_type == 'binary':
        # Binary: 0 for benign traffic, 1 for any attack
        df['Label_numeric'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    else:
        # Multiclass: Encode each attack type as a unique integer
        label_encoder = LabelEncoder()
        df['Label_numeric'] = label_encoder.fit_transform(df['Label'])
        df['Label_numeric'] = df['Label_numeric'].astype('int64')
        
        # Save the label encoder for later use in predictions
        pd.to_pickle(label_encoder, os.path.join(output_dir, f'{prefix}_label_encoder.pkl'))
    
    # Feature selection based on correlation with target
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
    correlation_matrix = df[numeric_columns].corr().abs()
    target_col = 'Label_numeric'
    
    # 1. Select features with correlation above threshold with target
    target_corr = correlation_matrix[target_col].abs()
    strong_features = target_corr[target_corr > correlation_threshold].index.tolist()
    
    # 2. Identify and remove highly correlated features (to reduce multicollinearity)
    reduced_corr_matrix = df[strong_features].corr()
    upper = reduced_corr_matrix.where(np.triu(np.ones(reduced_corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column].abs() > 0.85)]
    
    # Final feature set after removing redundant features
    final_features = [feat for feat in strong_features if feat not in to_drop]
    print("Final selected features after removing redundancy:", final_features)
    selected_df = df[final_features]

    # Apply ensemble outlier detection and removal
    filtered_df = outlier_removal(selected_df)
    
    # Save the processed dataset to CSV for reproducibility
    filtered_df.to_csv(os.path.join(output_dir, f'{prefix}_processed.csv'), index=False)
    print(f"Processed dataset saved to {os.path.join(output_dir, f'{prefix}_processed.csv')}")
    
    # Print summary statistics about the preprocessing
    final_shape = filtered_df.shape
    rows_removed_total = initial_shape[0] - final_shape[0]
    columns_removed_total = initial_shape[1] - final_shape[1]
    print(f"Final Dataset Shape: {final_shape}")
    print(f"Summary: {rows_removed_total} rows ({rows_removed_total/initial_shape[0]*100:.2f}%) and "
          f"{columns_removed_total} columns ({columns_removed_total/initial_shape[1]*100:.2f}%) removed in total.")

    # Log processing time
    stop_time = datetime.datetime.now()
    print(f"Time taken to pre-process the dataset: {stop_time - start_time}")
    
    # Return the processed dataframe for further pipeline stages
    return filtered_df

def outlier_removal(df):
    """
    Identify and remove outliers using an ensemble of three methods.
    
    Uses three different outlier detection algorithms and only keeps data points
    that all three methods agree are normal (not outliers). This conservative
    approach helps ensure we only remove true anomalies.
    
    Parameters:
    df (DataFrame): DataFrame with selected features and target variable
    
    Returns:
    DataFrame: Filtered dataset with outliers removed
    """
    # Separate the target variable from the dataset
    label_column = df["Label_numeric"]
    df_without_label = df.drop(columns=["Label_numeric"])

    # 1. Local Outlier Factor (LOF) - detects outliers based on local density
    # Good for detecting outliers that might be normal in global distribution
    model_lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    pred_lof = model_lof.fit_predict(df_without_label)

    # 2. Isolation Forest - isolates outliers through random splits
    # Especially effective for high-dimensional data
    model_iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    pred_if = model_iforest.fit_predict(df_without_label)

    # 3. Robust Covariance (Elliptic Envelope) - assumes Gaussian distribution
    # Works well for detecting outliers in normally distributed features
    model_robcov = EllipticEnvelope(contamination=0.05, random_state=42, support_fraction=0.9)
    pred_rc = model_robcov.fit_predict(df_without_label)

    # Add outlier detection results to dataframe (1 = inlier, -1 = outlier)
    df_without_label["pred_lof"] = pred_lof
    df_without_label["pred_if"] = pred_if
    df_without_label["pred_rc"] = pred_rc

    # Only keep data points that all three methods agree are NOT outliers
    filtered_data = df_without_label[
    (df_without_label["pred_lof"] == 1) & 
    (df_without_label["pred_if"] == 1) & 
    (df_without_label["pred_rc"] == 1)]

    # Remove the temporary prediction columns
    filtered_data = filtered_data.drop(columns=["pred_lof", "pred_if", "pred_rc"])
    
    # Reattach the target column (using the index to align correctly)
    filtered_data["Label_numeric"] = label_column.loc[filtered_data.index]

    return filtered_data

def create_train_test_split(df, output_dir, prefix, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets and apply scaling and PCA.
    
    This function:
    1. Splits data while preserving class distribution
    2. Standardizes features
    3. Applies PCA for dimensionality reduction
    4. Saves preprocessing models for later use in production
    
    Parameters:
    df (DataFrame): Processed dataset
    output_dir (str): Directory to save preprocessing artifacts
    prefix (str): Prefix for saved files
    test_size (float): Proportion of data to use for testing (default: 0.2)
    random_state (int): Random seed for reproducibility
    
    Returns:
    tuple: (X_train_pca, X_test_pca, y_train, y_test) - Processed data ready for model training
    """
    start_time = datetime.datetime.now()
    
    # Separate features and target
    X = df.drop('Label_numeric', axis=1)
    y = df['Label_numeric']
    
    # Split the data, using stratification to preserve label distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Apply StandardScaler to normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for use in production
    pd.to_pickle(scaler, os.path.join(output_dir, f'{prefix}_scaler.pkl'))
    
    # Apply PCA for dimensionality reduction
    n_components = min(4, X_train_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Save the PCA model for use in production
    pd.to_pickle(pca, os.path.join(output_dir, f'{prefix}_pca.pkl'))
    
    print(f"Train-test split saved to {output_dir}")
    print(f"Data shapes - X_train: {X_train_pca.shape}, X_test: {X_test_pca.shape}")
    
    # Uncomment to debug class distribution issues
    # print("Train Label Distribution:")
    # print(y_train.value_counts(normalize=True))
    # print("\nTest Label Distribution:")
    # print(y_test.value_counts(normalize=True))

    stop_time = datetime.datetime.now()
    print(f"Time taken to split and transform the dataset: {stop_time - start_time}")
    
    return X_train_pca, X_test_pca, y_train, y_test

if __name__ == "__main__":
    # Process three different datasets for different classification problems
    
    # Binary classification - attack vs. benign
    print("Processing Binary - 2 class dataset")
    input_path_1 = r"../../data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    output_dir_1 = r"../../data/preprocessed"
    df_1 = process_dataset(input_path_1, output_dir_1, 'friday_afternoon', 'binary', correlation_threshold=0.2)
    create_train_test_split(df_1, output_dir_1, 'friday_afternoon')

    # 3-class classification - Tuesday dataset with multiple attack types
    print("")
    print("Processing Multiclass - 3 class dataset")
    input_path_2 = r"../../data/raw/Tuesday-WorkingHours.pcap_ISCX.csv"
    output_dir_2 = r"../../data/preprocessed"
    df_2 = process_dataset(input_path_2, output_dir_2, 'tuesday_working', 'multiclass_3')
    create_train_test_split(df_2, output_dir_2, 'tuesday_working')

    # 4-class classification - Thursday dataset with web attacks
    print("")
    print("Processing Multiclass - 4 class dataset")
    input_path_3 = r"../../data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    output_dir_3 = r"../../data/preprocessed"
    df_3 = process_dataset(input_path_3, output_dir_3, 'thursday_morning', 'multiclass_4')
    create_train_test_split(df_3, output_dir_3, 'thursday_morning')