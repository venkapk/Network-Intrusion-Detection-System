import pickle
import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from src.preprocessing.data_preprocessor import process_dataset, create_train_test_split

def train_models():
    """
    Main entry point for training all models.
    Trains binary and multiple multiclass models based on different datasets.
    Creates directory structure and saves all models and evaluation metrics.
    """
    start_time = datetime.datetime.now()
    
    # Create directory structure for models and evaluation results
    os.makedirs('models/binary', exist_ok=True)
    os.makedirs('models/multiclass_3', exist_ok=True)
    os.makedirs('models/multiclass_4', exist_ok=True)
    os.makedirs('models/binary/evaluation', exist_ok=True)
    os.makedirs('models/multiclass_3/evaluation', exist_ok=True)
    os.makedirs('models/multiclass_4/evaluation', exist_ok=True)
    
    # Train binary classification (attack vs. benign) using Friday afternoon dataset
    # Ensure that correlation_threshold is 0.3 for binary classification and comment
    # 3-class and 4-class models while running binary
    input_path_binary = r"data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    output_dir_binary = r"data/preprocessed"
    prefix_binary = 'friday_afternoon'
    train_classification_models(input_path_binary, output_dir_binary, prefix_binary,
                  class_type = 'binary')
    
    # Train 3-class model using Tuesday dataset
    input_path_mul_3 = r"data/raw/Tuesday-WorkingHours.pcap_ISCX.csv"
    output_dir_mul_3 = r"data/preprocessed"
    prefix_mul_3 = 'tuesday_working'
    train_classification_models(input_path_mul_3, output_dir_mul_3, prefix_mul_3,
                                class_type = 'multiclass_3')
    
    # Train 4-class model using Thursday dataset
    input_path_mul_4 = r"data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    output_dir_mul_4 = r"data/preprocessed"
    prefix_mul_4 = 'thursday_morning'
    train_classification_models(input_path_mul_4, output_dir_mul_4, prefix_mul_4,
                                class_type = 'multiclass_4')

    # Print total execution time
    stop_time = datetime.datetime.now()
    print(f"Time taken to train all the models: {stop_time - start_time}")
    
def train_classification_models(input_path, output_dir, prefix, class_type = 'binary'):
    """
    Trains multiple classification models for a specific dataset and class type.
    
    The function:
    1. Loads and preprocesses the dataset
    2. Applies SMOTE for class balancing
    3. Performs data leakage testing
    4. Trains and tunes multiple model types (LR, RF, XGBoost)
    5. Evaluates and compares model performance
    6. Analyzes feature importance
    7. Generates diagnostic reports
    
    Parameters:
    -----------
    input_path : str
        Path to the raw CSV dataset
    output_dir : str
        Directory to save preprocessed data
    prefix : str
        Prefix for saved files
    class_type : str
        Type of classification problem: 'binary', 'multiclass_3', or 'multiclass_4'
    
    Returns:
    --------
    dict
        Dictionary containing diagnostic information about model training
    """
    start_time = datetime.datetime.now()
    print(f"Training {class_type} classification models...")
    
    # Load and preprocess data
    # Change correlation_threshold to 0.3 for binary classification
    df = process_dataset(input_path, output_dir, prefix, class_type,
                         correlation_threshold = 0.1)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = create_train_test_split(df, output_dir, prefix)
    
    # Apply SMOTE to handle class imbalance
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Original class distribution: {pd.Series(y_train).value_counts(normalize=True)}")
    print(f"Balanced class distribution: {pd.Series(y_train_balanced).value_counts(normalize=True)}")

    # Run sanity checks for data leakage
    print("\nRunning leakage tests with simple models...")
    leakage_results = test_for_leakage_with_simple_models(X_train_balanced, X_test, y_train_balanced,
                                                          y_test, class_type)

    # Train three different model types with hyperparameter tuning
    print("Training Logistic Regression model...")
    logistic_model = train_logistic_regression(X_train_balanced, X_test, y_train_balanced, y_test)
    
    print("Training Random Forest model...")
    rf_model = train_random_forest(X_train_balanced, X_test, y_train_balanced, y_test)
    
    print("Training XGBoost model...")
    xgb_model = train_xg_boost(X_train_balanced, X_test, y_train_balanced, y_test)
    
    # Evaluate all models on test data and generate visualizations
    print("Evaluating model performance...")
    evaluate_models(
        {
            'Logistic Regression': logistic_model,
            'Random Forest': rf_model,
            'XGBoost': xgb_model
        },
        X_test, y_test, class_type
    )

    # Analyze most important features for predictions
    print("\nAnalyzing feature importance in detail...")
    importance_results = detailed_feature_importance_analysis(
        {
            'Random Forest': rf_model,
            'XGBoost': xgb_model
        },
        X_test, class_type
    )
    
    # Analyze how features are distributed across different classes
    print("\nAnalyzing feature distributions across classes...")
    top_features = set()
    for model_results in importance_results.values():
        top_5_features = model_results['top_features'][:5]  # Get top 5 features
        top_features.update(top_5_features)
    
    feature_stats = analyze_feature_distributions(X_train_balanced, y_train_balanced,
                                                  list(top_features), class_type)
    
    # Save trained models for later deployment
    print("Saving models...")
    pickle.dump(logistic_model, open(f'models/{class_type}/logistic.pkl', 'wb'))
    pickle.dump(rf_model, open(f'models/{class_type}/random_forest.pkl', 'wb'))
    pickle.dump(xgb_model, open(f'models/{class_type}/xgboost.pkl', 'wb'))
    
    # Generate comprehensive report about model training and potential issues
    print("\nGenerating diagnostic summary report...")
    with open(f'models/{class_type}/evaluation/diagnostic_summary.txt', 'w') as f:
        f.write("="*50 + "\n")
        f.write("MODEL EVALUATION DIAGNOSTIC SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        # Summarize simple model tests
        f.write("SIMPLE MODEL TEST RESULTS:\n")
        f.write("-"*50 + "\n")
        for model, metrics in leakage_results.items():
            if model != 'Single Feature Tests':
                f.write(f"{model}:\n")
                for metric, value in metrics.items():
                    if metric in ['accuracy', 'f1_score']:
                        f.write(f"  {metric}: {value:.4f}\n")
                    else:
                        f.write(f"  {metric}: {value}\n")
        
        # Single feature tests
        f.write("\nSINGLE FEATURE TEST RESULTS:\n")
        f.write("-"*50 + "\n")
        for feature, metrics in leakage_results.get('Single Feature Tests', {}).items():
            f.write(f"Feature: {feature}\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  F1 Score: {metrics['f1_score']:.4f}\n")
        
        # Feature importance summary
        f.write("\nFEATURE IMPORTANCE SUMMARY:\n")
        f.write("-"*50 + "\n")
        for model, results in importance_results.items():
            f.write(f"{model} Top Features:\n")
            for i, (feature, importance) in enumerate(zip(results['top_features'][:5], 
                                                      results['top_importances'][:5])):
                f.write(f"  {i+1}. {feature}: {importance:.4f}\n")
            f.write(f"  Features needed for 95% importance: {results['features_for_95_pct']}\n\n")
        
        # Data leakage conclusion
        f.write("\nDATA LEAKAGE ASSESSMENT:\n")
        f.write("-"*50 + "\n")
        
        # Check if any single feature test had >95% accuracy
        high_accuracy_features = [f for f, m in leakage_results.get('Single Feature Tests', {}).items() 
                                if m['accuracy'] > 0.95]
        
        if high_accuracy_features:
            f.write("POTENTIAL DATA LEAKAGE DETECTED!\n")
            f.write(f"The following features achieve >95% accuracy alone, suggesting data leakage:\n")
            for feature in high_accuracy_features:
                acc = leakage_results['Single Feature Tests'][feature]['accuracy']
                f.write(f"  - {feature}: {acc:.2%} accuracy\n")
            f.write("\nRecommendation: Review these features for potential information leakage.\n")
        elif leakage_results['Decision Stump (depth=1)']['accuracy'] > 0.95:
            f.write("POTENTIAL DATA LEAKAGE DETECTED!\n")
            f.write(f"A decision stump (depth=1) achieves {leakage_results['Decision Stump (depth=1)']['accuracy']:.2%} accuracy.\n")
            f.write(f"The feature '{leakage_results['Decision Stump (depth=1)']['feature_used']}' may be leaking information.\n\n")
        else:
            f.write("No immediate evidence of data leakage detected from simple model tests.\n")
            f.write("However, the perfect performance of Random Forest and XGBoost still warrants investigation.\n\n")
    
    print(f"\n{class_type} classification model training and diagnostics complete.")
    stop_time = datetime.datetime.now()
    print(f"Time taken: {stop_time - start_time}")
    
    # Return diagnostics for potential further analysis
    return {
        'leakage_results': leakage_results,
        'importance_results': importance_results,
        'feature_stats': feature_stats
    }

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train and tune a Logistic Regression model.
    
    Uses grid search to find optimal hyperparameters with F1 as the scoring metric.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
        
    Returns:
    --------
    sklearn.linear_model.LogisticRegression
        Best tuned model
    """
    # Initialize model with reasonable defaults
    logistic_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    
    # Define hyperparameter search space
    param_grid = {
        'C': [0.01, 0.1, 1, 10],  # Regularization strength (inverse)
        'penalty': ['l2']         # L2 regularization
    }
    
    # Use grid search for hyperparameter tuning
    grid_search = GridSearchCV(estimator=logistic_model, param_grid=param_grid,
                                cv=3, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model from search
    best_logistic_model = grid_search.best_estimator_
    print(f"Best Logistic Regression parameters: {grid_search.best_params_}")
    print(f"Best Logistic Regression CV score: {grid_search.best_score_:.4f}")

    return best_logistic_model

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train and tune a Random Forest Classifier.
    
    Uses randomized search to efficiently explore hyperparameter space.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
        
    Returns:
    --------
    sklearn.ensemble.RandomForestClassifier
        Best tuned model
    """
    # Initialize RF with class_weight to handle any remaining class imbalance
    rf_model = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=10  # Use 10 cores for training
    )

    # Define hyperparameter search space
    param_dist = {
        'n_estimators': [100, 200, 500],           # Number of trees
        'max_depth': [None, 10, 20, 30],           # Max tree depth
        'min_samples_split': [2, 5, 10],           # Min samples to split node 
        'min_samples_leaf': [1, 2, 4],             # Min samples in leaf node
        'max_features': ['sqrt', 'log2'],          # Features to consider per split
        'bootstrap': [True, False]                 # Whether to bootstrap samples
    }
    
    # RandomizedSearchCV is faster than GridSearch for large parameter spaces
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='f1_weighted',
        random_state=42,
        n_jobs=10  # Parallelize across 10 cores
    )
    random_search.fit(X_train, y_train)
    
    # Get the best model
    best_rf_model = random_search.best_estimator_
    print(f"Best Random Forest parameters: {random_search.best_params_}")
    print(f"Best Random Forest CV score: {random_search.best_score_:.4f}")

    return best_rf_model

def train_xg_boost(X_train, X_test, y_train, y_test):
    """
    Train and tune an XGBoost Classifier.
    
    Uses grid search to find optimal hyperparameters and automatically 
    handles binary vs multiclass objectives.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
        
    Returns:
    --------
    xgboost.XGBClassifier
        Best tuned model
    """
    # Initialize XGBoost with different objective based on problem type
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax' if len(set(y_train)) > 2 else 'binary:logistic',
        num_class=len(set(y_train)) if len(set(y_train)) > 2 else None,
        random_state=42,
        eval_metric='mlogloss' if len(set(y_train)) > 2 else 'logloss'
    )
    
    # Define hyperparameter grid for XGBoost
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],       # Step size shrinkage
        'max_depth': [3, 5, 7],                  # Max tree depth
        'n_estimators': [100, 200],              # Number of trees
        'subsample': [0.8, 1.0],                 # Fraction of samples for trees
        'colsample_bytree': [0.8, 1.0],          # Fraction of features for trees
        'gamma': [0, 1]                          # Min loss reduction for split
    }
    
    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1  # Use all available cores
    )
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_xgb_model = grid_search.best_estimator_
    print(f"Best XGBoost parameters: {grid_search.best_params_}")
    print(f"Best XGBoost CV score: {grid_search.best_score_:.4f}")
    
    return best_xgb_model

def evaluate_models(models, X_test, y_test, class_type):
    """
    Evaluate multiple models and save evaluation metrics and visualizations.
    
    Creates confusion matrices, ROC curves, precision-recall curves, and
    comprehensive performance reports.
    
    Parameters:
    -----------
    models : dict
        Dictionary containing model name as key and trained model as value
    X_test : DataFrame
        Test features
    y_test : Series
        Test labels
    class_type : str
        Type of classification ('binary', 'multiclass_3', or 'multiclass_4')
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics for each model
    """
    results = {}
    is_binary = (len(set(y_test)) == 2)
    
    for name, model in models.items():
        # Generate predictions and probability scores
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Calculate standard classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Binary and multiclass metrics are handled differently
        if is_binary:
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            avg_precision = average_precision_score(y_test, y_prob[:, 1])
        else:
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            avg_precision = 'N/A'
        
        # Create and save confusion matrix visualization
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(y_test)), yticklabels=sorted(set(y_test)))
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'models/{class_type}/evaluation/{name}_confusion_matrix.png')
        plt.close()
        
        # For binary classification, create ROC curve
        if is_binary:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend(loc='lower right')
            plt.savefig(f'models/{class_type}/evaluation/{name}_roc_curve.png')
            plt.close()
            
            # Create Precision-Recall curve for binary classification
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob[:, 1])
            
            plt.figure(figsize=(10, 8))
            plt.plot(recall_curve, precision_curve, color='blue', lw=2, 
                     label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {name}')
            plt.legend(loc='lower left')
            plt.savefig(f'models/{class_type}/evaluation/{name}_pr_curve.png')
            plt.close()
        
        # Save detailed classification report
        report = classification_report(y_test, y_pred)
        with open(f'models/{class_type}/evaluation/{name}_classification_report.txt', 'w') as f:
            f.write(report)
        
        # Store results for comparison
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_precision': avg_precision
        }
    
    # Save model comparison results to CSV
    comparison_df = pd.DataFrame(results).T.round(4)
    comparison_df.to_csv(f'models/{class_type}/evaluation/model_comparison.csv')
    
    # Create bar chart comparing model performance
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    plt.figure(figsize=(12, 8))
    comparison_df[metrics].plot(kind='bar', figsize=(12, 8))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=0)
    plt.ylim([0, 1])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'models/{class_type}/evaluation/model_comparison.png')
    plt.close()
    
    # Print summary results
    print("\nModel Performance Comparison:")
    print(comparison_df)
    
    # Determine best model based on F1 score
    best_model = comparison_df['f1_score'].idxmax()
    print(f"\nBest model based on F1 score: {best_model}")
    
    return results

def detailed_feature_importance_analysis(models, X_test, class_type):
    """
    Perform detailed analysis of feature importance for tree-based models.
    
    Identifies the most important features, checks for potential data leakage
    (single dominant features), and determines how many features are needed 
    for 95% of the total importance.
    
    Parameters:
    -----------
    models : dict
        Dictionary of tree-based models (Random Forest, XGBoost)
    X_test : array-like
        Test features
    class_type : str
        Type of classification task
        
    Returns:
    --------
    dict
        Dictionary with feature importance analysis results for each model
    """
    # Check if X_test is a DataFrame or numpy array
    is_dataframe = hasattr(X_test, 'columns')
    
    if is_dataframe:
        feature_names = X_test.columns.tolist()
    else:
        # Create generic feature names if X_test is a numpy array
        feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
    
    importance_results = {}
    
    for name, model in models.items():
        if name in ['Random Forest', 'XGBoost']:
            # Extract feature importances - both RF and XGBoost have this attribute
            if name == 'Random Forest':
                importances = model.feature_importances_
            elif name == 'XGBoost':
                importances = model.feature_importances_
            
            # Sort feature importances in descending order
            indices = np.argsort(importances)[::-1]
            
            # Calculate cumulative importance
            sorted_importances = [importances[i] for i in indices]
            sorted_features = [feature_names[i] for i in indices]
            cumulative_importance = np.cumsum(sorted_importances)
            
            # Print the top features for this model
            print(f"\n=== {name} Feature Importance Analysis ===")
            for i in range(10):
                if i < len(indices):
                    feature_idx = indices[i]
                    print(f"Feature {i+1}: {feature_names[feature_idx]} - {importances[feature_idx]:.4f}")
            
            # Check for potential data leakage
            if sorted_importances[0] > 0.5:  # Single feature explains >50% of variance
                print(f"WARNING: Feature '{sorted_features[0]}' explains {sorted_importances[0]:.2%} of importance.")
                print("This could indicate data leakage or a proxy for the target variable.")
            
            # Check how many features needed for 95% importance
            features_for_95 = np.argmax(cumulative_importance >= 0.95) + 1
            print(f"Number of features needed for 95% importance: {features_for_95} out of {len(feature_names)}")
            
            # Save detailed feature importances to DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances,
                'Cumulative_Importance': np.nan
            })
            
            # Sort by importance and add cumulative importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()
            
            # Create directory if it doesn't exist
            os.makedirs(f'models/{class_type}/evaluation', exist_ok=True)
            
            # Save to CSV
            importance_df.to_csv(f'models/{class_type}/evaluation/{name}_detailed_feature_importance.csv', index=False)
            
            # Store results for return
            importance_results[name] = {
                'top_features': sorted_features[:10],
                'top_importances': sorted_importances[:10],
                'features_for_95_pct': features_for_95
            }
    
    return importance_results


def test_for_leakage_with_simple_models(X_train, X_test, y_train, y_test, class_type):
    """
    Test for potential data leakage using extremely simple models.
    
    If simple models like decision stumps (depth=1) or single features
    achieve very high accuracy, it suggests data leakage or a feature
    that's too predictive (potentially from the future).
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
    class_type : str
        Type of classification task
        
    Returns:
    --------
    dict
        Results of leakage tests
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    
    results = {}
    
    # Check if X_train is a DataFrame or numpy array
    is_dataframe = hasattr(X_train, 'columns')
    
    # Get feature names (if available)
    if is_dataframe:
        feature_names = X_train.columns.tolist()
    else:
        # Create generic feature names if X_train is a numpy array
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    # Test with decision stumps (depth=1)
    # A stump is the simplest possible tree with just one split
    # If this gets high accuracy, there's likely a problem
    dt_stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    dt_stump.fit(X_train, y_train)
    y_pred_stump = dt_stump.predict(X_test)
    stump_accuracy = accuracy_score(y_test, y_pred_stump)
    stump_f1 = f1_score(y_test, y_pred_stump, average='weighted')
    
    # Get the feature used by the stump
    feature_idx = dt_stump.tree_.feature[0]
    # Only get feature name if it's not -2 (which indicates a leaf node)
    feature_name = feature_names[feature_idx] if feature_idx != -2 else "None (Pure leaf)"
    threshold = dt_stump.tree_.threshold[0] if feature_idx != -2 else "N/A"
    
    results['Decision Stump (depth=1)'] = {
        'accuracy': stump_accuracy,
        'f1_score': stump_f1,
        'feature_used': feature_name,
        'threshold': threshold
    }
    
    print(f"\n=== Decision Stump (depth=1) Results ===")
    print(f"Accuracy: {stump_accuracy:.4f}")
    print(f"F1 Score: {stump_f1:.4f}")
    print(f"Feature Used: {feature_name}")
    print(f"Split Threshold: {threshold}")
    
    # Test with slightly more complex tree (depth=2)
    dt_2 = DecisionTreeClassifier(max_depth=2, random_state=42)
    dt_2.fit(X_train, y_train)
    y_pred_dt2 = dt_2.predict(X_test)
    dt2_accuracy = accuracy_score(y_test, y_pred_dt2)
    dt2_f1 = f1_score(y_test, y_pred_dt2, average='weighted')
    
    results['Decision Tree (depth=2)'] = {
        'accuracy': dt2_accuracy,
        'f1_score': dt2_f1
    }
    
    print(f"\n=== Decision Tree (depth=2) Results ===")
    print(f"Accuracy: {dt2_accuracy:.4f}")
    print(f"F1 Score: {dt2_f1:.4f}")
    
    # Test individual features with simple logistic regression
    print("\n=== Testing Individual Features with Logistic Regression ===")
    
    # Sort features by importance with a simple decision tree
    dt_full = DecisionTreeClassifier(random_state=42)
    dt_full.fit(X_train, y_train)
    importances = dt_full.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Test top 5 features individually
    single_feature_results = {}
    for i in range(min(5, len(indices))):
        feature_idx = indices[i]
        feature_name = feature_names[feature_idx]
        
        # Extract the single feature for training/testing
        if is_dataframe:
            X_train_single = X_train.iloc[:, feature_idx].values.reshape(-1, 1)
            X_test_single = X_test.iloc[:, feature_idx].values.reshape(-1, 1)
        else:
            X_train_single = X_train[:, feature_idx].reshape(-1, 1)
            X_test_single = X_test[:, feature_idx].reshape(-1, 1)
        
        # Train a simple logistic regression on this single feature
        lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
        lr.fit(X_train_single, y_train)
        y_pred_single = lr.predict(X_test_single)
        
        single_accuracy = accuracy_score(y_test, y_pred_single)
        single_f1 = f1_score(y_test, y_pred_single, average='weighted')
        
        single_feature_results[feature_name] = {
            'accuracy': single_accuracy,
            'f1_score': single_f1
        }
        
        print(f"Feature: {feature_name}")
        print(f"  Accuracy: {single_accuracy:.4f}")
        print(f"  F1 Score: {single_f1:.4f}")
        
        # Warning if single feature gives nearly perfect performance
        if single_accuracy > 0.95:
            print(f"  WARNING: Single feature '{feature_name}' achieves {single_accuracy:.2%} accuracy!")
            print("  This strongly indicates data leakage.")
    
    results['Single Feature Tests'] = single_feature_results
    
    # Save all leakage test results to CSV
    results_df = pd.DataFrame({
        'Model': ['Decision Stump (depth=1)', 'Decision Tree (depth=2)'] + 
                 [f"Single Feature: {f}" for f in single_feature_results.keys()],
        'Accuracy': [results['Decision Stump (depth=1)']['accuracy'], 
                    results['Decision Tree (depth=2)']['accuracy']] + 
                    [v['accuracy'] for v in single_feature_results.values()],
        'F1 Score': [results['Decision Stump (depth=1)']['f1_score'], 
                    results['Decision Tree (depth=2)']['f1_score']] + 
                    [v['f1_score'] for v in single_feature_results.values()]
    })
    
    os.makedirs(f'models/{class_type}/evaluation', exist_ok=True)
    results_df.to_csv(f'models/{class_type}/evaluation/leakage_test_results.csv', index=False)
    
    return results

def analyze_feature_distributions(X_train, y_train, important_features, class_type):
    """
    Analyze how features are distributed across different classes.
    
    Creates histograms showing feature distributions by class and checks
    for perfect separation, which could indicate data leakage.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    important_features : list
        List of feature names to analyze
    class_type : str
        Type of classification task
        
    Returns:
    --------
    dict
        Feature statistics by class
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create output directory for feature distribution analysis
    os.makedirs(f'models/{class_type}/evaluation/feature_distributions', exist_ok=True)
    
    # Check if X_train is a DataFrame
    is_dataframe = hasattr(X_train, 'columns')
    
    if is_dataframe:
        # Use pandas directly if it's a DataFrame
        analysis_df = X_train.copy()
        # Convert y_train to Series if it's not already
        if not isinstance(y_train, pd.Series):
            if isinstance(y_train, np.ndarray):
                y_train = pd.Series(y_train)
            else:
                y_train = pd.Series(np.array(y_train))
        analysis_df['target'] = y_train.values
    else:
        # Create DataFrame from numpy arrays
        data_dict = {}
        for i, feature_name in enumerate(important_features):
            if i < X_train.shape[1]:  # Make sure index is valid
                data_dict[feature_name] = X_train[:, i]
        
        # Add target column
        data_dict['target'] = y_train
        analysis_df = pd.DataFrame(data_dict)
    
    # Analyze feature statistics by class
    feature_stats = {}
    
    for feature in important_features:
        # Skip if feature doesn't exist in the DataFrame
        if feature not in analysis_df.columns:
            continue
            
        # Create histogram of feature values separated by class
        plt.figure(figsize=(12, 6))
        sns.histplot(data=analysis_df, x=feature, hue='target', kde=True, common_norm=False)
        plt.title(f'Distribution of {feature} by Class')
        plt.savefig(f'models/{class_type}/evaluation/feature_distributions/{feature}_hist.png')
        plt.close()
        
        # Calculate statistics by class
        stats_by_class = analysis_df.groupby('target')[feature].agg(['mean', 'std', 'min', 'max'])
        feature_stats[feature] = stats_by_class
        
        # Check for perfect separation between classes
        class_ranges = {}
        for cls in analysis_df['target'].unique():
            class_data = analysis_df[analysis_df['target'] == cls][feature]
            class_ranges[cls] = (class_data.min(), class_data.max())
        
        # Check if the ranges overlap
        ranges_overlap = False
        for cls1 in class_ranges:
            for cls2 in class_ranges:
                if cls1 != cls2:
                    min1, max1 = class_ranges[cls1]
                    min2, max2 = class_ranges[cls2]
                    if (min1 <= max2 and max1 >= min2):
                        ranges_overlap = True
                        break
        
        # Warn about perfectly separating features
        if not ranges_overlap:
            print(f"WARNING: Feature '{feature}' perfectly separates classes with no overlap!")
            print(f"Class ranges: {class_ranges}")
    
    # Save statistics to CSV
    for feature, stats in feature_stats.items():
        stats.to_csv(f'models/{class_type}/evaluation/feature_distributions/{feature}_stats.csv')
    
    return feature_stats

if __name__ == "__main__":
    train_models()