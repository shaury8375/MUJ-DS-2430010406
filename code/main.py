"""Knowledge distillation on CIFAR-10: Teacher vs Student vs Distilled Student.

Usage:
    python code/main.py
    python code/main.py --epochs 10 --temperature 4.0 --alpha 0.5

Plots are saved to the results/ folder.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import StudentCNN, TeacherCNN, count_parameters
from train import (evaluate_model, measure_inference_time,
                   train_distilled_student, train_model)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def get_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = torchvision.datasets.CIFAR10("./data", train=True,
                                             download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10("./data", train=False,
                                            download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    return train_loader, test_loader


def save_plots(results_dir, names, accuracies, params, curves):
    os.makedirs(results_dir, exist_ok=True)

    # Test accuracy comparison
    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, accuracies)
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy Comparison")
    plt.ylim(min(accuracies) - 5, 100)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{acc:.2f}%", ha="center")
    plt.savefig(os.path.join(results_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()

    # Model size comparison
    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, params)
    plt.ylabel("Number of Parameters")
    plt.title("Model Size Comparison")
    plt.yscale("log")
    for bar, count in zip(bars, params):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{count:,}", ha="center", va="bottom")
    plt.savefig(os.path.join(results_dir, "model_size_comparison.png"), dpi=150)
    plt.close()

    # Training accuracy curves
    plt.figure(figsize=(10, 6))
    for name, curve in zip(names, curves):
        plt.plot(curve, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy (%)")
    plt.title("Teacher vs Student vs Distilled Student")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_dir, "training_curves.png"), dpi=150)
    plt.close()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, test_loader = get_data(args.batch_size)

    # 1. Teacher
    print("\n--- Training Teacher ---")
    teacher = TeacherCNN().to(device)
    _, teacher_curve = train_model(teacher, train_loader, device,
                                   args.epochs, args.lr)
    teacher_acc = evaluate_model(teacher, test_loader, device)

    # 2. Normal student
    print("\n--- Training Normal Student ---")
    student = StudentCNN().to(device)
    _, student_curve = train_model(student, train_loader, device,
                                   args.epochs, args.lr)
    student_acc = evaluate_model(student, test_loader, device)

    # 3. Distilled student
    print("\n--- Training Distilled Student ---")
    distilled = StudentCNN().to(device)
    _, distilled_curve = train_distilled_student(
        distilled, teacher, train_loader, device, args.epochs, args.lr,
        args.temperature, args.alpha)
    distilled_acc = evaluate_model(distilled, test_loader, device)

    # Summary
    names = ["Teacher", "Normal Student", "Distilled Student"]
    accuracies = [teacher_acc, student_acc, distilled_acc]
    params = [count_parameters(m) for m in (teacher, student, distilled)]
    times = [measure_inference_time(m, test_loader, device)
             for m in (teacher, student, distilled)]

    print("\n" + "=" * 70)
    print("                    KNOWLEDGE DISTILLATION")
    print("=" * 70)
    print(f"{'Model':<20}{'Accuracy':<12}{'Parameters':<16}{'Inference (s)'}")
    print("-" * 70)
    for n, a, p, t in zip(names, accuracies, params, times):
        print(f"{n:<20}{a:<12.2f}{p:<16,}{t:.4f}")
    print("=" * 70)
    print(f"Compression ratio: {params[0] / params[1]:.2f}x")

    save_plots(args.results_dir, names, accuracies, params,
               [teacher_curve, student_curve, distilled_curve])
    print(f"\nPlots saved to '{args.results_dir}/'")


if __name__ == "__main__":
    main()
