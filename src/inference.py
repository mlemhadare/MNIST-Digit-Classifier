"""
inference.py
Inference pipeline for MNIST digit classification.
"""
from typing import Tuple
import torch
import numpy as np
from model import MNISTCNN
from torchvision import transforms

def predict_image(model: torch.nn.Module, image_array: np.ndarray, device: torch.device = None) -> Tuple[int, float]:
    """
    Predict the digit and confidence score for a single image.

    Args:
        model (torch.nn.Module): Trained MNISTCNN model.
        image_array (np.ndarray): Raw image array (28x28, uint8 or float32).
        device (torch.device, optional): Device to run inference on.

    Returns:
        Tuple[int, float]: Predicted digit and confidence score.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    if image_array.shape != (28, 28):
        raise ValueError("Input image must be 28x28 pixels.")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image_array).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item()
