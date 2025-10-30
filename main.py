from datetime import datetime
import json
import numpy as np
from pathlib import Path

import torch
from dataset.gait_preprocessor import prepare_dataloaders
from metrics.model_analyzer import ModelAnalyzer
from model.gait_classifier import GaitClassifier
from model.gait_cnn import GaitCNN
from model.gait_tcn import GaitTCN
from optimizer.hyper_param_optimizer import create_best_model, print_study_results, run_optuna_study
from optimizer.tcn import run_optuna_study_tcn, print_study_results_tcn

def run_training_pipeline():
    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/run_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")
    
    # Configuration
    config = {
        'data_files': 'data/TrainingDataV2/*.csv',
        'window_length': 100,
        'stride': 50,
        'batch_size': 32,
        'label_column': 'gait_type',
        'test_size': 0.2,
        'random_state': 42,
        'epochs': 50,
        'learning_rate': 0.001,
        'early_stopping_patience': 10,
        'class_names': ['Stand', 'Walk', 'Jog']
    }
    
    # Save config
    with open(log_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("========Loading and preparing data========")
    train_loader, test_loader, preprocessor = prepare_dataloaders(
        data_files=config['data_files'],
        window_length=config['window_length'],
        stride=config['stride'],
        batch_size=config['batch_size'],
        label_column=config['label_column'],
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
    print("\n========Creating model========")
    sample_features, _ = next(iter(train_loader))
    num_channels = sample_features.shape[2]
    
    # model = GaitCNN(
    #    input_length=config['window_length'],
    #    input_channels=num_channels,
    #    num_classes=len(config['class_names'])
    #)

    model = GaitTCN(
        input_length=config['window_length'],
        input_channels=num_channels,
        num_classes=len(config['class_names']),
        num_channels=[32, 64, 128],
        kernel_size=3,
        dropout_rate=0.5
    )
    
    print("\n========Creating classifier========")
    classifier = GaitClassifier(
        model=model,
        class_names=config['class_names']
    )
    
    print("\n========Training model========")
    history = classifier.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        early_stopping_patience=config['early_stopping_patience'],
        save_best_model=True,
        model_save_path=str(log_dir / 'best_model.pth')
    )
    
    print("\n========Final evaluation========")
    test_results = classifier.evaluate(test_loader, return_predictions=True)
    
    # Save test results
    with open(log_dir / 'test_results.json', 'w') as f:
        results_to_save = {
            'loss': float(test_results['loss']),
            'accuracy': float(test_results['accuracy']),
            'class_accuracies': {k: float(v) for k, v in test_results['class_accuracies'].items()}
        }
        json.dump(results_to_save, f, indent=4)
    
    print("\n========Example predictions========")
    sample_batch, sample_labels = next(iter(test_loader))
    predictions = classifier.predict_with_names(sample_batch[:5])
    
    print("\nSample Predictions:")
    for i, (class_name, confidence) in enumerate(predictions):
        true_label = config['class_names'][sample_labels[i].item()]
        correct = "✓" if class_name == true_label else "✗"
        print(f"  {correct} Sample {i+1}: {class_name} ({confidence:.1f}% confidence) | True: {true_label}")
    
    print("\n========Generating comprehensive analysis========")
    analyzer = ModelAnalyzer(
        classifier=classifier,
        class_names=config['class_names'],
        device=classifier.device
    )
    analyzer.generate_report(test_loader, log_dir)
    
    print("\n========Saving final model========")
    classifier.save_model(str(log_dir / 'final_model.pth'))
    
    print("\n" + "="*60)
    print("✓ Training pipeline complete!")
    print("="*60)
    print(f"\n📁 Results saved to: {log_dir}/")
    print("\n📊 Generated files:")
    print("  - config.json")
    print("  - best_model.pth")
    print("  - final_model.pth")
    print("  - test_results.json")
    print("  - training_curves.png")
    print("  - confusion_matrix.png")
    print("  - roc_curves.png")
    print("  - confidence_distribution.png")
    print("  - misclassifications.csv")
    print("  - training_history.csv")
    print("  - classification_report.txt")
    print(f"\n{'='*60}\n")
    
    return classifier, analyzer, log_dir

def run_tcn_training_pipeline():
    """
    Complete training pipeline for TCN gait classification model.
    """
    
    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/tcn_run_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"TCN Training Pipeline")
    print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")
    
    # Configuration
    config = {
        # Data parameters
        'data_files': 'data/TrainingDataV4/*.csv',
        'window_length': 100,
        'stride': 50,
        'batch_size': 32,
        'label_column': 'gait_type',
        'test_size': 0.2,
        'random_state': 42,
        
        # Model parameters
        'num_channels': [64, 128, 256],
        'kernel_size': 7,
        'dropout_rate': 0.3,
        
        # Training parameters
        'epochs': 50,
        'learning_rate': 0.001,
        'early_stopping_patience': 10,
        
        # Class information
        'class_names': ['Stand', 'Walk', 'Jog']
    }
    
    # Save config
    with open(log_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Load and prep data
    print("========Loading and preparing data========")
    train_loader, test_loader, preprocessor = prepare_dataloaders(
        data_files=config['data_files'],
        window_length=config['window_length'],
        stride=config['stride'],
        batch_size=config['batch_size'],
        label_column=config['label_column'],
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
    # Create Model
    print("\n========Creating TCN model========")
    sample_features, _ = next(iter(train_loader))
    num_channels = sample_features.shape[2]
    
    model = GaitTCN(
        input_length=config['window_length'],
        input_channels=num_channels,
        num_classes=len(config['class_names']),
        num_channels=config['num_channels'],
        kernel_size=config['kernel_size'],
        dropout_rate=config['dropout_rate']
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create Classifier
    print("\n========Creating classifier========")
    classifier = GaitClassifier(
        model=model,
        class_names=config['class_names']
    )
    
    # Train TCN
    print("\n========Training model========")
    history = classifier.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        early_stopping_patience=config['early_stopping_patience'],
        save_best_model=True,
        model_save_path=str(log_dir / 'best_model.pth')
    )
    
    # Evaluate Model Performance
    print("\n========Final evaluation========")
    test_results = classifier.evaluate(test_loader, return_predictions=True)
    
    # Save test results
    with open(log_dir / 'test_results.json', 'w') as f:
        results_to_save = {
            'loss': float(test_results['loss']),
            'accuracy': float(test_results['accuracy']),
            'class_accuracies': {k: float(v) for k, v in test_results['class_accuracies'].items()}
        }
        json.dump(results_to_save, f, indent=4)
    
    # Sample predictions
    print("\n========Example predictions========")
    sample_batch, sample_labels = next(iter(test_loader))
    predictions = classifier.predict_with_names(sample_batch[:5])
    
    print("\nSample Predictions:")
    for i, (class_name, confidence) in enumerate(predictions):
        true_label = config['class_names'][sample_labels[i].item()]
        correct = "✓" if class_name == true_label else "✗"
        print(f"  {correct} Sample {i+1}: {class_name} ({confidence:.1f}% confidence) | True: {true_label}")
    
    # Analysis
    print("\n========Generating comprehensive analysis========")
    analyzer = ModelAnalyzer(
        classifier=classifier,
        class_names=config['class_names'],
        device=classifier.device
    )
    analyzer.generate_report(test_loader, log_dir)
    
    # Save Model
    print("\n========Saving final model========")
    classifier.save_model(str(log_dir / 'final_model.pth'))
    
    # Print summary
    print("\n" + "="*60)
    print("✓ TCN Training pipeline complete!")
    print("="*60)
    print(f"\n📁 Results saved to: {log_dir}/")
    print("\n📊 Generated files:")
    print("  - config.json")
    print("  - best_model.pth")
    print("  - final_model.pth")
    print("  - test_results.json")
    print("  - training_curves.png")
    print("  - confusion_matrix.png")
    print("  - roc_curves.png")
    print("  - confidence_distribution.png")
    print("  - misclassifications.csv")
    print("  - training_history.csv")
    print("  - classification_report.txt")
    print(f"\n{'='*60}\n")
    
    return classifier, analyzer, log_dir

def hyperparam_optim_tcn():
    print("="*60)
    print("GAIT TCN HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    # Configuration
    config = {
        'data_files': 'data/TrainingDataV2/*.csv',
        'window_length': 100,
        'stride': 50,
        'batch_size': 32,
        'label_column': 'gait_type',
        'test_size': 0.2,
        'random_state': 42,
    }
    
    print("\n1. Preparing data...")
    train_loader, val_loader, preprocessor = prepare_dataloaders(
        data_files=config['data_files'],
        window_length=config['window_length'],
        stride=config['stride'],
        batch_size=config['batch_size'],
        label_column=config['label_column'],
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
    # Convert DataLoaders to numpy arrays for Optuna
    X_train, y_train = [], []
    for features, labels in train_loader:
        X_train.append(features.numpy())
        y_train.append(labels.numpy())
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    X_val, y_val = [], []
    for features, labels in val_loader:
        X_val.append(features.numpy())
        y_val.append(labels.numpy())
    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)
    
    num_classes = len(np.unique(y_train))
    
    print(f"✓ Training data: {X_train.shape}")
    print(f"✓ Validation data: {X_val.shape}")
    print(f"✓ Number of classes: {num_classes}")
    
    # Run Optuna optimization for TCN
    # Now automatically detects input_channels from X_train.shape[2]
    print("\n2. Running Optuna optimization for TCN...")
    
    study = run_optuna_study_tcn(
        X_train, y_train, 
        X_val, y_val,
        num_classes=num_classes,
        n_trials=30,
        n_epochs=15,
        study_name='gait_tcn_tuning',
        use_pruning=True
    )
    
    best_params = print_study_results_tcn(study)

if __name__ == "__main__":
    #run_training_pipeline()
    #hyperparam_optim()
    #hyperparam_optim_tcn()
    run_tcn_training_pipeline()