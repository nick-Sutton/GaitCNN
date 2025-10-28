from datetime import datetime
import json
from pathlib import Path
from dataset.gait_preprocessor import prepare_dataloaders
from metrics.model_analyzer import ModelAnalyzer
from model.gait_classifier import GaitClassifier
from model.gait_cnn import GaitCNN

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
    
    model = GaitCNN(
        input_length=config['window_length'],
        input_channels=num_channels,
        num_classes=len(config['class_names'])
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

if __name__ == "__main__":
    run_training_pipeline()