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
        optimizer, mode='max', factor=0.5, patience=3, verbose=False
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


# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    
    print("="*60)
    print("GAIT CNN HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("="*60)
    
    # ==========================================
    # 1. PREPARE YOUR DATA
    # ==========================================
    
    print("\n1. Preparing data...")
    
    # Example: Create dummy data (replace with your actual data)
    n_samples_train = 1000
    n_samples_val = 200
    time_steps = 100
    n_channels = 24
    num_classes = 3
    
    # Simulate data (replace with your actual normalized data)
    X_train = np.random.rand(n_samples_train, time_steps, n_channels).astype(np.float32)
    y_train = np.random.randint(0, num_classes, n_samples_train)
    
    X_val = np.random.rand(n_samples_val, time_steps, n_channels).astype(np.float32)
    y_val = np.random.randint(0, num_classes, n_samples_val)
    
    print(f"✓ Training data: {X_train.shape}")
    print(f"✓ Validation data: {X_val.shape}")
    print(f"✓ Number of classes: {num_classes}")
    
    # ==========================================
    # 2. RUN OPTUNA OPTIMIZATION
    # ==========================================
    
    print("\n2. Running Optuna optimization...")
    print("   This may take a while depending on n_trials and n_epochs...")
    
    study = run_optuna_study(
        X_train, y_train, 
        X_val, y_val,
        num_classes=num_classes,
        n_trials=20,      # Start with 20 trials (increase for better results)
        n_epochs=15,      # 15 epochs per trial
        study_name='gait_cnn_tuning'
    )
    
    # ==========================================
    # 3. PRINT RESULTS
    # ==========================================
    
    best_params = print_study_results(study)
    
    # ==========================================
    # 4. CREATE AND SAVE BEST MODEL
    # ==========================================
    
    print("\n" + "="*60)
    print("CREATING BEST MODEL")
    print("="*60)
    
    best_model = create_best_model(
        best_params, 
        num_classes=num_classes,
        input_length=time_steps,
        input_channels=n_channels
    )
    
    print(f"✓ Best model created")
    print(f"✓ Total parameters: {sum(p.numel() for p in best_model.parameters()):,}")
    
    # Save best model
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'hyperparameters': best_params,
        'num_classes': num_classes
    }, 'best_gait_cnn_model.pth')
    
    print(f"✓ Model saved to 'best_gait_cnn_model.pth'")
    
    # ==========================================
    # 5. VISUALIZE OPTIMIZATION (Optional)
    # ==========================================
    
    print("\n" + "="*60)
    print("VISUALIZATION (if you have plotly installed)")
    print("="*60)
    
    try:
        import optuna.visualization as vis
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html('optimization_history.html')
        print("✓ Saved optimization_history.html")
        
        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html('param_importances.html')
        print("✓ Saved param_importances.html")
        
        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html('parallel_coordinate.html')
        print("✓ Saved parallel_coordinate.html")
        
    except ImportError:
        print("⚠ Install plotly to visualize results: pip install plotly")
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review the best hyperparameters above")
    print("  2. Train final model with best_params on full dataset")
    print("  3. Check visualization HTML files for insights")
    print("  4. Consider running more trials for better results")