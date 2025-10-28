from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_fscore_support,
    roc_curve, 
    auc
)
from sklearn.preprocessing import label_binarize

from dataset.gait_preprocessor import prepare_dataloaders
from model.gait_classifier import GaitClassifier
from model.gait_cnn import GaitCNN


class ModelAnalyzer:
    """Comprehensive model analysis for gait classification"""
    
    def __init__(self, classifier, class_names, device):
        self.classifier = classifier
        self.model = classifier.model
        self.class_names = class_names
        self.device = device
        
    def get_predictions_and_probabilities(self, data_loader):
        """Get predictions, probabilities, and true labels"""
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs), np.array(all_labels)
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=True, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_display = cm_norm * 100
            fmt = '.1f'
            title = 'Confusion Matrix (%)'
        else:
            cm_display = cm
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Percentage' if normalize else 'Count'})
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        if normalize:
            for i in range(len(self.class_names)):
                for j in range(len(self.class_names)):
                    plt.text(j + 0.5, i + 0.7, f'(n={cm[i,j]})',
                            ha='center', va='center', fontsize=9, color='gray')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_roc_curves(self, y_true, y_probs, save_path=None):
        """Plot ROC curves for each class"""
        n_classes = len(self.class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        if n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, color=color,
                    label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def analyze_misclassifications(self, y_true, y_pred, y_probs, top_n=20, save_path=None):
        """Analyze most confident misclassifications"""
        misclass_indices = np.where(y_true != y_pred)[0]
        
        if len(misclass_indices) == 0:
            print("✓ No misclassifications - Perfect accuracy!")
            return pd.DataFrame()
        
        misclass_data = []
        for idx in misclass_indices:
            true_class = self.class_names[y_true[idx]]
            pred_class = self.class_names[y_pred[idx]]
            confidence = y_probs[idx, y_pred[idx]]
            true_prob = y_probs[idx, y_true[idx]]
            
            misclass_data.append({
                'sample_index': idx,
                'true_class': true_class,
                'predicted_class': pred_class,
                'confidence_%': confidence * 100,
                'true_class_prob_%': true_prob * 100,
                'margin_%': (confidence - true_prob) * 100
            })
        
        df = pd.DataFrame(misclass_data)
        df = df.sort_values('confidence_%', ascending=False).head(top_n)
        
        print(f"\n{'='*80}")
        print(f"TOP {min(top_n, len(df))} MOST CONFIDENT MISCLASSIFICATIONS")
        print(f"{'='*80}")
        print(df.to_string(index=False, float_format='%.2f'))
        print(f"{'='*80}\n")
        
        if save_path:
            df.to_csv(save_path, index=False, float_format='%.4f')
        
        return df
    
    def plot_confidence_distribution(self, y_true, y_pred, y_probs, save_path=None):
        """Plot confidence distribution for correct vs incorrect predictions"""
        correct_mask = y_true == y_pred
        
        if not np.any(~correct_mask):
            print("✓ All predictions correct!")
            return
        
        correct_conf = np.max(y_probs[correct_mask], axis=1)
        incorrect_conf = np.max(y_probs[~correct_mask], axis=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(correct_conf, bins=30, alpha=0.7, label='Correct', 
                     color='green', edgecolor='black')
        axes[0].hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', 
                     color='red', edgecolor='black')
        axes[0].set_xlabel('Confidence', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot
        bp = axes[1].boxplot([correct_conf, incorrect_conf], 
                             labels=['Correct', 'Incorrect'],
                             patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        axes[1].set_ylabel('Confidence', fontsize=12)
        axes[1].set_title('Confidence Comparison', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        print(f"Confidence - Correct: {np.mean(correct_conf):.3f}, "
              f"Incorrect: {np.mean(incorrect_conf):.3f}")
    
    def plot_training_curves(self, history, save_path=None):
        """Plot training history with loss and accuracy"""
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train', markersize=3)
        axes[0].plot(epochs, history['val_loss'], 'r-o', label='Val', markersize=3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Mark best epoch
        best_epoch = np.argmin(history['val_loss'])
        axes[0].axvline(x=best_epoch+1, color='red', linestyle='--', alpha=0.5)
        
        # Accuracy
        axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train', markersize=3)
        axes[1].plot(epochs, history['val_acc'], 'g-o', label='Val', markersize=3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy', fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Overfitting indicator
        loss_ratio = np.array(history['val_loss']) / np.array(history['train_loss'])
        axes[2].plot(epochs, loss_ratio, 'purple', marker='o', markersize=3)
        axes[2].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Val/Train Loss Ratio')
        axes[2].set_title('Overfitting Indicator', fontweight='bold')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_report(self, test_loader, save_dir):
        """Generate complete analysis report"""
        print("\n========Generating analysis report========")
        
        # Get predictions
        test_preds, test_probs, test_labels = self.get_predictions_and_probabilities(test_loader)
        
        # 1. Classification report
        print("\nClassification Report:")
        report = classification_report(test_labels, test_preds, 
                                      target_names=self.class_names, digits=3)
        print(report)
        with open(save_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        # 2. Training curves
        self.plot_training_curves(self.classifier.history,
                                 save_path=save_dir / 'training_curves.png')
        
        # 3. Confusion matrix
        self.plot_confusion_matrix(test_labels, test_preds, normalize=True,
                                   save_path=save_dir / 'confusion_matrix.png')
        
        # 4. ROC curves
        self.plot_roc_curves(test_labels, test_probs,
                            save_path=save_dir / 'roc_curves.png')
        
        # 5. Confidence distribution
        self.plot_confidence_distribution(test_labels, test_preds, test_probs,
                                         save_path=save_dir / 'confidence_distribution.png')
        
        # 6. Misclassifications
        self.analyze_misclassifications(test_labels, test_preds, test_probs, top_n=20,
                                       save_path=save_dir / 'misclassifications.csv')
        
        # 7. Save training history
        pd.DataFrame(self.classifier.history).to_csv(save_dir / 'training_history.csv', index=False)
        
        print(f"\n✓ Analysis complete! Results saved to: {save_dir}/")