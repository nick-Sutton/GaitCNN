import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

class GaitClassifier:
    """
    Wrapper class for GaitCNN that handles training, evaluation, and inference.
    """
    
    def __init__(self, model, device=None, class_names=None):
        """
        Args:
            model: GaitCNN model instance
            device: 'cuda', 'cpu', or None (auto-detect)
            class_names: Optional list of class names for readable output
        """
        self.model = model
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Class names for readable predictions
        self.class_names = class_names or [f"Class_{i}" for i in range(model.num_classes)]
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def train(self, 
              train_loader, 
              val_loader, 
              epochs=50,
              learning_rate=0.001,
              weight_decay=1e-5,
              early_stopping_patience=10,
              save_best_model=True,
              model_save_path='best_model.pth'):
        """
        Train the model with early stopping and model checkpointing.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Maximum number of epochs
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization strength
            early_stopping_patience: Stop if no improvement for N epochs
            save_best_model: Whether to save the best model
            model_save_path: Path to save the best model
        
        Returns:
            Dictionary with training history
        """
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("\n" + "="*60)
        print("TRAINING START")
        print("="*60)
        print(f"Epochs: {epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print("="*60 + "\n")
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping and model checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_best_model:
                    self.save_model(model_save_path)
                    print(f"  → Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        
        return self.history
    
    def _train_epoch(self, train_loader, criterion, optimizer):
        """Single training epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in tqdm(train_loader, desc="Training", leave=False):
            features, labels = features.to(self.device), labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate_epoch(self, val_loader, criterion):
        """Single validation epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc="Validation", leave=False):
                features, labels = features.to(self.device), labels.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, test_loader, return_predictions=False):
        """
        Evaluate model on test set and optionally return predictions.
        
        Args:
            test_loader: DataLoader for test data
            return_predictions: If True, return predictions and true labels
        
        Returns:
            Dictionary with metrics (and optionally predictions)
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        print("\nEvaluating model...")
        
        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc="Testing"):
                features, labels = features.to(self.device), labels.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                running_loss += loss.item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        test_loss = running_loss / len(test_loader)
        test_acc = 100 * np.mean(all_predictions == all_labels)
        
        # Per-class accuracy
        unique_classes = np.unique(all_labels)
        class_accuracies = {}
        for cls in unique_classes:
            mask = all_labels == cls
            class_acc = 100 * np.mean(all_predictions[mask] == all_labels[mask])
            class_name = self.class_names[cls]
            class_accuracies[class_name] = class_acc
        
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print("\nPer-Class Accuracy:")
        for class_name, acc in class_accuracies.items():
            print(f"  {class_name}: {acc:.2f}%")
        print("="*60)
        
        results = {
            'loss': test_loss,
            'accuracy': test_acc,
            'class_accuracies': class_accuracies
        }
        
        if return_predictions:
            results['predictions'] = all_predictions
            results['true_labels'] = all_labels
            results['probabilities'] = all_probabilities
        
        return results
    
    def predict(self, features, return_probabilities=False):
        """
        Make predictions on new data.
        
        Args:
            features: Tensor or numpy array of shape (batch, window_length, channels)
            return_probabilities: If True, return class probabilities
        
        Returns:
            predictions (and optionally probabilities)
        """
        self.model.eval()
        
        # Convert to tensor if needed
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        features = features.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        predicted = predicted.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        
        if return_probabilities:
            return predicted, probabilities
        return predicted
    
    def predict_with_names(self, features):
        """
        Make predictions and return class names with confidence scores.
        
        Args:
            features: Tensor or numpy array
        
        Returns:
            List of tuples (class_name, confidence)
        """
        predictions, probabilities = self.predict(features, return_probabilities=True)
        
        results = []
        for pred, probs in zip(predictions, probabilities):
            class_name = self.class_names[pred]
            confidence = probs[pred] * 100
            results.append((class_name, confidence))
        
        return results
    
    def plot_training_history(self, save_path=None):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, path='model.pth'):
        """Save model weights and configuration"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_length': self.model.input_length,
                'input_channels': self.model.input_channels,
                'num_classes': self.model.num_classes
            },
            'class_names': self.class_names,
            'history': self.history
        }, path)
        
    def load_model(self, path='model.pth'):
        """Load model weights and configuration"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_names = checkpoint.get('class_names', self.class_names)
        self.history = checkpoint.get('history', self.history)
        print(f"Model loaded from {path}")