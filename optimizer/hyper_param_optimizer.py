import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model.gait_classifier import GaitClassifier
from model.gait_cnn import GaitCNN

# Import your model classes
# from your_module import GaitClassificationCNN, GaitClassifier, DataNormalizer, calculate_class_weights


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
    
    try:
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
    except Exception as e:
        print(f"Error creating model: {e}")
        raise optuna.TrialPruned()
    
    # Create classifier wrapper
    classifier = GaitClassifier(model, device=device)
    
    # ==========================================
    # 3. CREATE DATA LOADERS
    # ==========================================
    
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size
    )
    
    # ==========================================
    # 4. TRAIN MODEL
    # ==========================================
    
    # Train using your GaitClassifier.train() method
    # We'll monitor validation accuracy and report to Optuna
    history = classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=5,
        save_best_model=False  # Don't save during optimization
    )
    
    # Get best validation accuracy from history
    best_val_acc = max(history['val_acc'])
    
    # Report final accuracy to Optuna
    return best_val_acc


def objective_with_pruning(trial, X_train, y_train, X_val, y_val, num_classes, 
                           input_length=100, input_channels=24, n_epochs=20):
    """
    Alternative objective with manual training loop for better Optuna pruning.
    This allows Optuna to stop unpromising trials early.
    """
    
    # 1. Suggest hyperparameters (same as above)
    w1 = trial.suggest_int('w1', 10, 30, step=5)
    f11 = trial.suggest_categorical('f11', [7, 9, 11, 13, 15])
    f12 = trial.suggest_categorical('f12', [3, 5, 7, 9])
    w2 = trial.suggest_int('w2', 10, 25, step=5)
    f21 = trial.suggest_categorical('f21', [7, 9, 11, 13])
    f22 = trial.suggest_categorical('f22', [7, 9, 11, 13])
    p11 = trial.suggest_categorical('p11', [2, 3])
    p12 = trial.suggest_categorical('p12', [2, 3])
    p21 = trial.suggest_categorical('p21', [2, 3])
    p22 = trial.suggest_categorical('p22', [2, 3])
    fc_neurons = trial.suggest_int('fc_neurons', 512, 4096, step=512)
    dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.7)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # 2. Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
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
        model = model.to(device)
    except Exception as e:
        print(f"Error creating model: {e}")
        raise optuna.TrialPruned()
    
    # 3. Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size)
    
    # 4. Training loop with Optuna pruning
    best_val_acc = 0.0
    
    for epoch in range(n_epochs):
        # Train
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Report to Optuna for pruning
        trial.report(val_acc, epoch)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_acc


def run_optuna_study(X_train, y_train, X_val, y_val, num_classes,
                     n_trials=50, n_epochs=20, study_name='gait_cnn_optimization',
                     use_pruning=True):
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
        use_pruning: If True, use objective with pruning (faster)
    
    Returns:
        study: Optuna study object with results
    """
    
    # Create Optuna study
    if use_pruning:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    else:
        pruner = optuna.pruners.NopPruner()
    
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # Maximize validation accuracy
        pruner=pruner
    )
    
    # Choose objective function
    if use_pruning:
        obj_func = lambda trial: objective_with_pruning(
            trial, X_train, y_train, X_val, y_val, 
            num_classes, n_epochs=n_epochs
        )
        print(f"Using objective WITH pruning (faster, stops bad trials early)")
    else:
        obj_func = lambda trial: objective(
            trial, X_train, y_train, X_val, y_val, 
            num_classes, n_epochs=n_epochs
        )
        print(f"Using objective WITHOUT pruning (uses your GaitClassifier.train())")
    
    # Run optimization
    study.optimize(
        obj_func,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1  # Use 1 for debugging, increase for parallel trials
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
    
    print(f"  Value (Validation Accuracy): {trial.value:.2f}%")
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