"""
main.py
Entry point for MNIST project: EDA, training, evaluation, and inference demo.
"""
from data import get_data_loaders, plot_sample_images
from train import run_optuna_study
from model import MNISTCNN
from inference import predict_image
from utils import visualize_predictions
import torch
import numpy as np

def main():
    # 1. Data loading and EDA
    train_loader, val_loader, test_loader = get_data_loaders("data", batch_size=64)
    print("Showing 10 random MNIST samples:")
    plot_sample_images(train_loader, n=10)

    # 2. Hyperparameter tuning
    print("Running Optuna hyperparameter search...")
    best_params = run_optuna_study()

    # 4. Inference demo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTCNN(dropout_rate=best_params["dropout"]).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # Demo: Predict a random test image
    images, labels = next(iter(test_loader))
    image = images[0][0].numpy()  # shape (28, 28)
    pred, conf = predict_image(model, image, device=device)
    print(f"Predicted: {pred}, Confidence: {conf:.2f}, True label: {labels[0].item()}")
    visualize_predictions(model, device, test_loader, num_images=25)


if __name__ == "__main__":
    main()
