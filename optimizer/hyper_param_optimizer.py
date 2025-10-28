import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model.gait_classifier import GaitClassifier
from model.gait_cnn import GaitCNN


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    """Create PyTorch DataLoaders"""
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for debugging, increase for speed
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader


def objective(trial, X_train, y_train, X_val, y_val, num_classes, 
              input_length=100, input_channels=24, n_epochs=20):
    """
    Optuna objective function for hyperparameter optimization
    
    Args:
        trial: Optuna trial object
        X_train: Training data (n_samples, time_steps, channels)
        y_train: Training labels (n_samples,)
        X_val: Validation data
        y_val: Validation labels
        num_classes: Number of classes to predict
        input_length: Time window length
        input_channels: Number of sensor channels
        n_epochs: Number of training epochs per trial
    
    Returns:
        Best validation accuracy achieved
    """
    
    # ==========================================
    # 1. SUGGEST HYPERPARAMETERS
    # ==========================================
    
    # Architecture hyperparameters
    # Conv1 parameters
    w1 = trial.suggest_int('w1', 10, 30, step=5)  # Number of filters: 10, 15, 20, 25, 30
    f11 = trial.suggest_categorical('f11', [7, 9, 11, 13, 15])  # Kernel height (odd)
    f12 = trial.suggest_categorical('f12', [3, 5, 7, 9])  # Kernel width (odd)
    
    # Conv2 parameters
    w2 = trial.suggest_int('w2', 10, 25, step=5)  # Number of filters
    f21 = trial.suggest_categorical('f21', [7, 9, 11, 13])
    f22 = trial.suggest_categorical('f22', [7, 9, 11, 13])
    
    # Pooling parameters (standard sizes)
    p11 = trial.suggest_categorical('p11', [2, 3])
    p12 = trial.suggest_categorical('p12', [2, 3])
    p21 = trial.suggest_categorical('p21', [2, 3])
    p22 = trial.suggest_categorical('p22', [2, 3])
    
    # Fully connected layer size
    fc_neurons = trial.suggest_int('fc_neurons', 512, 4096, step=512)
    
    # Dropout rate
    dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.7)
    
    # Training hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # ==========================================
    # 2. CREATE MODEL
    # ==========================================
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GaitCNN(
        input_length=input_length,
        input_channels=input_channels,
        num_classes=num_classes,
        f11=f11, f12=f12, w1=w1,
        p11=p11, p12=p12,
        f21=f21, f22=f22, w2=w2,
        p21=p21, p22=p22,
        fc_neurons=fc_neurons,
        dropout_rate=dropout_rate
    )
    
    # Calculate class weights if data is imbalanced
    #class_weights = calculate_class_weights(y_train, num_classes)
    
    classifier = GaitClassifier(
        model, 
        device=device
    )
    
    # ==========================================
    # 3. SETUP TRAINING
    # ==========================================
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler (optional but recommended)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size
    )
    
    # ==========================================
    # 4. TRAINING LOOP
    # ==========================================
    
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 5
    
    for epoch in range(n_epochs):
        # Training
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        model.train()
        for batch_data, batch_labels in train_loader:
            loss, acc = classifier.train_step(batch_data, batch_labels, optimizer)
            train_loss += loss * batch_data.size(0)
            train_correct += acc * batch_data.size(0)
            train_total += batch_data.size(0)
        
        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total
        
        # Validation
        val_loss, val_acc = classifier.validate(val_loader)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Report intermediate value to Optuna
        trial.report(val_acc, epoch)
        
        # Early stopping (for Optuna pruning)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping (for this trial)
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_acc


def run_optuna_study(X_train, y_train, X_val, y_val, num_classes,
                     n_trials=50, n_epochs=20, study_name='gait_cnn_optimization'):
    """
    Run Optuna hyperparameter optimization study
    
    Args:
        X_train: Training data (n_samples, time_steps, channels)
        y_train: Training labels (n_samples,)
        X_val: Validation data
        y_val: Validation labels
        num_classes: Number of classes
        n_trials: Number of Optuna trials to run
        n_epochs: Number of epochs per trial
        study_name: Name for the study
    
    Returns:
        study: Optuna study object with results
    """
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # Maximize validation accuracy
        pruner=optuna.pruners.MedianPruner(  # Prune unpromising trials
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, X_val, y_val, 
            num_classes, n_epochs=n_epochs
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    return study


def print_study_results(study):
    """Print optimization results"""
    print("\n" + "="*60)
    print("OPTUNA OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    print("\n" + "-"*60)
    print("BEST TRIAL")
    print("-"*60)
    trial = study.best_trial
    
    print(f"  Value (Validation Accuracy): {trial.value:.4f}")
    print(f"\n  Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    print("\n" + "-"*60)
    print("TOP 5 TRIALS")
    print("-"*60)
    df = study.trials_dataframe().sort_values('value', ascending=False).head(5)
    print(df[['number', 'value', 'params_learning_rate', 'params_batch_size', 
              'params_w1', 'params_w2', 'params_fc_neurons']])
    
    return trial.params


def create_best_model(best_params, num_classes, input_length=100, input_channels=24):
    """Create model with best hyperparameters from Optuna"""
    model = GaitCNN(
        input_length=input_length,
        input_channels=input_channels,
        num_classes=num_classes,
        f11=best_params['f11'],
        f12=best_params['f12'],
        w1=best_params['w1'],
        p11=best_params['p11'],
        p12=best_params['p12'],
        f21=best_params['f21'],
        f22=best_params['f22'],
        w2=best_params['w2'],
        p21=best_params['p21'],
        p22=best_params['p22'],
        fc_neurons=best_params['fc_neurons'],
        dropout_rate=best_params['dropout_rate']
    )
    
    return model