from dataset.gait_preprocessor import prepare_dataloaders
from model.gait_classifier import GaitClassifier
from model.gait_cnn import GaitCNN

def run_training_pipleine():
    print("========Loading and preparing data========")
    train_loader, test_loader, preprocessor = prepare_dataloaders(
        data_files='data/TrainingDataV2/*.csv',
        window_length=100,
        stride=50,
        batch_size=32,
        label_column='gait_type',
        test_size=0.2,
        random_state=42
    )
    
    print("\n========Creating model========")
    # Get number of channels from a sample batch
    sample_features, _ = next(iter(train_loader))
    num_channels = sample_features.shape[2]
    
    model = GaitCNN(
        input_length=100,
        input_channels=num_channels,
        num_classes=3,
    )
    
    print("\n========Creating classifier========")
    classifier = GaitClassifier(
        model=model,
        class_names=['Stand', 'Walk', 'Jog'] 
    )
    
    print("\n========Training model========")
    history = classifier.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=50,
        learning_rate=0.001,
        early_stopping_patience=10,
        save_best_model=True,
        model_save_path='best_gait_model.pth'
    )
    
    print("\n========Plotting training history========")
    classifier.plot_training_history(save_path='training_history.png')
    
    print("\n========Final evaluation========")
    test_results = classifier.evaluate(test_loader, return_predictions=True)
    
    print("\n========Example predictions========")
    sample_batch, _ = next(iter(test_loader))
    predictions = classifier.predict_with_names(sample_batch[:5])
    
    print("\nSample Predictions:")
    for i, (class_name, confidence) in enumerate(predictions):
        print(f"  Sample {i+1}: {class_name} ({confidence:.1f}% confidence)")
    
    print("\n========Saving final model========")
    classifier.save_model('final_gait_model.pth')
    
    print("\n✓ Training pipeline complete!")

if __name__ == "__main__":
    run_training_pipleine()