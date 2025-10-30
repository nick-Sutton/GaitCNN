import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model.gait_classifier import GaitClassifier
from model.gait_tcn import GaitTCN


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
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader


def objective_tcn(trial, X_train, y_train, X_val, y_val, num_classes, 
                  input_length=100, input_channels=None, n_epochs=20):
    """
    Optuna objective function for TCN hyperparameter optimization
    
    Args:
        trial: Optuna trial object
        X_train: Training data (n_samples, time_steps, channels)
        y_train: Training labels (n_samples,)
        X_val: Validation data
        y_val: Validation labels
        num_classes: Number of classes to predict
        input_length: Time window length
        input_channels: Number of sensor channels (auto-detected if None)
        n_epochs: Number of training epochs per trial
    
    Returns:
        Best validation accuracy achieved
    """
    
    # Auto-detect input channels if not provided
    if input_channels is None:
        input_channels = X_train.shape[2]
    """
    Optuna objective function for TCN hyperparameter optimization
    
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
    # 1. SUGGEST TCN HYPERPARAMETERS
    # ==========================================
    
    # Architecture: Number of TCN layers
    num_layers = trial.suggest_int('num_layers', 3, 6)
    
    # Channel progression strategy
    channel_strategy = trial.suggest_categorical('channel_strategy', 
                                                  ['constant', 'linear', 'exponential'])
    
    # Base number of channels
    base_channels = trial.suggest_int('base_channels', 32, 128, step=16)
    
    # Build channel list based on strategy
    if channel_strategy == 'constant':
        # All layers same size: [64, 64, 64, 64]
        num_channels = [base_channels] * num_layers
    elif channel_strategy == 'linear':
        # Linear growth: [32, 48, 64, 80]
        num_channels = [base_channels + i * 16 for i in range(num_layers)]
    else:  # exponential
        # Exponential growth: [32, 64, 128, 256]
        num_channels = [base_channels * (2 ** (i // 2)) for i in range(num_layers)]
        # Cap maximum channels to prevent memory issues
        num_channels = [min(ch, 256) for ch in num_channels]
    
    # Kernel size (critical for receptive field)
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7, 9])
    
    # Dropout rate
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Training hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # ==========================================
    # 2. CREATE TCN MODEL
    # ==========================================
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = GaitTCN(
            input_length=input_length,
            input_channels=input_channels,
            num_classes=num_classes,
            num_channels=num_channels,
            kernel_size=kernel_size,
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
    
    history = classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=5,
        save_best_model=False
    )
    
    # Get best validation accuracy from history
    best_val_acc = max(history['val_acc'])
    
    return best_val_acc


def objective_tcn_with_pruning(trial, X_train, y_train, X_val, y_val, num_classes, 
                                input_length=100, input_channels=26, n_epochs=20):
    """
    TCN objective with manual training loop for Optuna pruning.
    Allows stopping unpromising trials early.
    """
    
    # 1. Suggest hyperparameters
    num_layers = trial.suggest_int('num_layers', 3, 6)
    channel_strategy = trial.suggest_categorical('channel_strategy', 
                                                  ['constant', 'linear', 'exponential'])
    base_channels = trial.suggest_int('base_channels', 32, 128, step=16)
    
    # Build channel list
    if channel_strategy == 'constant':
        num_channels = [base_channels] * num_layers
    elif channel_strategy == 'linear':
        num_channels = [base_channels + i * 16 for i in range(num_layers)]
    else:  # exponential
        num_channels = [base_channels * (2 ** (i // 2)) for i in range(num_layers)]
        num_channels = [min(ch, 256) for ch in num_channels]
    
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7, 9])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # 2. Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = GaitTCN(
            input_length=input_length,
            input_channels=input_channels,
            num_classes=num_classes,
            num_channels=num_channels,
            kernel_size=kernel_size,
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


def run_optuna_study_tcn(X_train, y_train, X_val, y_val, num_classes,
                         n_trials=50, n_epochs=20, study_name='gait_tcn_optimization',
                         use_pruning=True):
    """
    Run Optuna hyperparameter optimization study for TCN
    
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
        obj_func = lambda trial: objective_tcn_with_pruning(
            trial, X_train, y_train, X_val, y_val, 
            num_classes, n_epochs=n_epochs
        )
        print(f"Using TCN objective WITH pruning (faster, stops bad trials early)")
    else:
        obj_func = lambda trial: objective_tcn(
            trial, X_train, y_train, X_val, y_val, 
            num_classes, n_epochs=n_epochs
        )
        print(f"Using TCN objective WITHOUT pruning (uses GaitClassifier.train())")
    
    # Run optimization
    study.optimize(
        obj_func,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1
    )
    
    return study


def print_study_results_tcn(study):
    """Print TCN optimization results"""
    print("\n" + "="*60)
    print("TCN OPTUNA OPTIMIZATION RESULTS")
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
    
    # Reconstruct the channel architecture
    num_layers = trial.params['num_layers']
    base_channels = trial.params['base_channels']
    channel_strategy = trial.params['channel_strategy']
    
    if channel_strategy == 'constant':
        num_channels = [base_channels] * num_layers
    elif channel_strategy == 'linear':
        num_channels = [base_channels + i * 16 for i in range(num_layers)]
    else:  # exponential
        num_channels = [base_channels * (2 ** (i // 2)) for i in range(num_layers)]
        num_channels = [min(ch, 256) for ch in num_channels]
    
    print(f"\n  Resulting TCN Architecture:")
    print(f"    Channel progression: {num_channels}")
    
    # Calculate receptive field
    kernel_size = trial.params['kernel_size']
    receptive_field = 1
    for i in range(num_layers):
        dilation = 2 ** i
        receptive_field += 2 * (kernel_size - 1) * dilation
    print(f"    Receptive field: {receptive_field} timesteps")
    
    print("\n" + "-"*60)
    print("TOP 5 TRIALS")
    print("-"*60)
    df = study.trials_dataframe().sort_values('value', ascending=False).head(5)
    print(df[['number', 'value', 'params_learning_rate', 'params_batch_size', 
              'params_num_layers', 'params_base_channels', 'params_kernel_size']])
    
    return trial.params
    