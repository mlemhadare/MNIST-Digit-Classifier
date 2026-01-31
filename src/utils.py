"""
utils.py
Utility functions: EarlyStopping, confusion matrix plotting, etc.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss does not improve.
    """
    def __init__(self, patience: int = 3) -> None:
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def plot_confusion_matrix(cm: np.ndarray, filename: str) -> None:
    """
    Plot and save a confusion matrix heatmap.
    Args:
        cm (np.ndarray): Confusion matrix.
        filename (str): Path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def visualize_predictions(model, device, test_loader, num_images=25):
    """
    Visualize predictions on test images with color-coded titles.
    
    Args:
        model (torch.nn.Module): Trained model.
        device (torch.device): Device for inference.
        test_loader (DataLoader): Test data loader.
        num_images (int): Number of images to display (default 25).
    """
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_images].to(device), labels[:num_images]
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Un-normalize images for display (MNIST mean/std)
    mean = 0.1307
    std = 0.3081
    images = images.cpu().numpy() * std + mean
    images = np.clip(images, 0, 1)  # Ensure [0,1] range
    
    # Plot grid
    rows = int(np.sqrt(num_images))
    cols = (num_images + rows - 1) // rows  # Ceiling division
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i][0], cmap='gray')
        ax.axis('off')
        
        true_label = labels[i].item()
        pred_label = preds[i].item()
        color = 'green' if pred_label == true_label else 'red'
        weight = 'bold' if pred_label != true_label else 'normal'
        ax.set_title(f'T: {true_label} | P: {pred_label}', color=color, fontweight=weight)
    
    plt.tight_layout()
    plt.show()

