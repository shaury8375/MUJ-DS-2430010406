"""Training, evaluation and distillation utilities."""

import time

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, device, epochs=10, learning_rate=0.001):
    """Standard supervised training. Returns per-epoch loss and accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses, accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}%")

    return losses, accuracies


def evaluate_model(model, test_loader, device):
    """Return test accuracy (%)."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def distillation_loss(student_outputs, teacher_outputs, labels,
                      temperature=4.0, alpha=0.5):
    """alpha * hard-label CE + (1 - alpha) * T^2 * KL(teacher || student)."""
    hard_loss = nn.CrossEntropyLoss()(student_outputs, labels)

    teacher_soft = torch.softmax(teacher_outputs / temperature, dim=1)
    student_log_soft = torch.log_softmax(student_outputs / temperature, dim=1)

    soft_loss = nn.KLDivLoss(reduction="batchmean")(student_log_soft, teacher_soft)
    soft_loss = soft_loss * (temperature ** 2)

    return alpha * hard_loss + (1 - alpha) * soft_loss


def train_distilled_student(student, teacher, train_loader, device, epochs=10,
                            learning_rate=0.001, temperature=4.0, alpha=0.5):
    """Train the student using the teacher's soft predictions."""
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    losses, accuracies = [], []
    teacher.eval()

    for epoch in range(epochs):
        student.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher(images)

            student_outputs = student(images)
            loss = distillation_loss(student_outputs, teacher_outputs, labels,
                                     temperature, alpha)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(student_outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}%")

    return losses, accuracies


def measure_inference_time(model, test_loader, device, num_batches=20):
    """Total time (seconds) to run the model on `num_batches` test batches."""
    model.eval()
    start = time.time()

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            model(images.to(device))

    if device.type == "cuda":
        torch.cuda.synchronize()

    return time.time() - start
