# MUJ-DS-2430010406

## Student Details
- **Name:** Shaury Sharma
- **Registration Number:** 2430010406
- **Branch:** Data Science & Engineering
- **Section:** D
- **GitHub Username:** shaury8375
# MUJ-DS-2430010406

## Student Details
- **Name:** Shaury Sharma
- **Registration Number:** 2430010406
- **Branch:** Data Science & Engineering
- **Section:** D
- **Batch:** F
- **GitHub Username:** shaury8375

## Training Program
- **Program:** Data Science / Deep Learning Training Program
- **Institution:** Manipal University Jaipur
- **Duration:** 2024 to 2028

## Project Title
**Knowledge Distillation for Model Compression: Teacher-Student CNNs on CIFAR-10**

## Project Summary
Large neural networks achieve high accuracy but are expensive to deploy. This project applies **knowledge distillation** to compress a large CNN (the *teacher*) into a much smaller CNN (the *student*) for image classification on the **CIFAR-10** dataset. Three models are trained and compared:

1. **Teacher CNN**: a deep CNN with BatchNorm and Dropout (~3.25M parameters)
2. **Normal Student CNN**: a small CNN trained only on the hard labels (~357K parameters)
3. **Distilled Student CNN**: the same small CNN trained with the teacher's soft predictions plus the true labels

The project is implemented in **PyTorch** and was trained on a Tesla T4 GPU in Google Colab.

## Dataset
- **CIFAR-10**: 60,000 colour images (32x32) in 10 classes
- 50,000 training images and 10,000 test images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Loaded automatically through `torchvision.datasets.CIFAR10`

## Methodology
- **Preprocessing:** images converted to tensors and normalized to the range [-1, 1]
- **Optimizer / loss:** Adam (lr = 0.001), cross-entropy, 10 epochs, batch size 128
- **Distillation loss:** a weighted sum of the hard-label cross-entropy and the KL-divergence between the softened teacher and student outputs
  - Temperature `T = 4.0`
  - Alpha `alpha = 0.5`
  - Soft loss scaled by `T^2`

```
Loss = alpha * CrossEntropy(student, labels)
     + (1 - alpha) * T^2 * KLDiv(softmax(student/T), softmax(teacher/T))
```

## Results

| Model             | Test Accuracy | Parameters |
|-------------------|---------------|------------|
| Teacher           | 83.79%        | 3,249,994  |
| Normal Student    | 75.28%        | 356,810    |
| Distilled Student | 76.49%        | 356,810    |

- **Compression ratio:** 9.11x fewer parameters (teacher to student)
- The distilled student improved on the normally trained student by about **1.2 percentage points** with the identical architecture.
- Inference time over 20 test batches: Teacher 1.17 s, Normal Student 1.06 s, Distilled Student 0.72 s (timings vary between runs).

Result plots (accuracy comparison, model size, training curves, sample images, prediction distribution) are in the `results/` folder and in the notebook.

## Repository Structure
- `assignments/`: Training assignments
- `notebooks/`: Jupyter/Colab notebook with the full experiment
- `code/`: Python source files (`models.py`, `train.py`, `main.py`, `requirements.txt`)
- `resources/`: Dataset links, references, requirements
- `presentations/`: Project presentation (`Knowledge_Distillation_Capstone.pptx`)
- `results/`: Result plots and screenshots
- `capstone/`: Capstone project work and links

## Installation and Usage

**Requirements:** Python 3.9+, and a GPU is recommended (CPU also works, but slowly).

```bash
git clone https://github.com/shaury8375/MUJ-DS-2430010406.git
cd MUJ-DS-2430010406
pip install -r code/requirements.txt
python code/main.py          # runs the full experiment, saves plots to results/
# or explore the notebook:
jupyter notebook notebooks/
```

Or open the notebook directly in **Google Colab**, select **Runtime → Change runtime type → T4 GPU**, and run all cells. The CIFAR-10 dataset downloads automatically on the first run.

## Tech Stack
Python, PyTorch, torchvision, NumPy, Matplotlib, Google Colab (Tesla T4 GPU)

## Capstone Repository
[Link to team capstone repo](https://github.com/ORG/REPO-NAME)

## Future Work
- Tune temperature and alpha for better distillation gains
- Train for more epochs and add data augmentation
- Try other compression methods such as pruning and quantization
- Deploy the distilled student on a lightweight web or mobile app

## Training Program
- **Program:** Data Science / Deep Learning Training Program
- **Institution:** Manipal University Jaipur
- **Duration:** <2024> to <2028>

## Project Title
**Knowledge Distillation for Model Compression: Teacher-Student CNNs on CIFAR-10**

## Project Summary
Large neural networks achieve high accuracy but are expensive to deploy. This project applies **knowledge distillation** to compress a large CNN (the *teacher*) into a much smaller CNN (the *student*) for image classification on the **CIFAR-10** dataset. Three models are trained and compared:

1. **Teacher CNN**: a deep CNN with BatchNorm and Dropout (~3.25M parameters)
2. **Normal Student CNN**: a small CNN trained only on the hard labels (~357K parameters)
3. **Distilled Student CNN**: the same small CNN trained with the teacher's soft predictions plus the true labels

The project is implemented in **PyTorch** and was trained on a Tesla T4 GPU in Google Colab.

## Dataset
- **CIFAR-10**: 60,000 colour images (32x32) in 10 classes
- 50,000 training images and 10,000 test images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Loaded automatically through `torchvision.datasets.CIFAR10`

## Methodology
- **Preprocessing:** images converted to tensors and normalized to the range [-1, 1]
- **Optimizer / loss:** Adam (lr = 0.001), cross-entropy, 10 epochs, batch size 128
- **Distillation loss:** a weighted sum of the hard-label cross-entropy and the KL-divergence between the softened teacher and student outputs
  - Temperature `T = 4.0`
  - Alpha `alpha = 0.5`
  - Soft loss scaled by `T^2`

```
Loss = alpha * CrossEntropy(student, labels)
     + (1 - alpha) * T^2 * KLDiv(softmax(student/T), softmax(teacher/T))
```

## Results

| Model             | Test Accuracy | Parameters |
|-------------------|---------------|------------|
| Teacher           | 83.79%        | 3,249,994  |
| Normal Student    | 75.28%        | 356,810    |
| Distilled Student | 76.49%        | 356,810    |

- **Compression ratio:** 9.11x fewer parameters (teacher to student)
- The distilled student improved on the normally trained student by about **1.2 percentage points** with the identical architecture.
- Inference time on a sample image: Teacher 1.17 s, Normal Student 1.06 s, Distilled Student 0.72 s.

Training curves, accuracy and model-size comparison charts, and sample predictions are in the notebook and in the `capstone/` folder.

## Repository Structure
- `assignments/`: Training assignments
- `notebooks/`: Jupyter/Colab notebook with the full experiment
- `code/`: Python source files
- `resources/`: Dataset links, references, requirements
- `presentations/`: Project presentation
- `capstone/`: Capstone project work and links

## Installation and Usage

**Requirements:** Python 3.9+, and a GPU is recommended (CPU also works, but slowly).

```bash
git clone https://github.com/<your-github-username>/MUJ-DS-2430010406.git
cd MUJ-DS-2430010406
pip install torch torchvision matplotlib numpy jupyter
jupyter notebook notebooks/
```

Or open the notebook directly in **Google Colab**, select **Runtime → Change runtime type → T4 GPU**, and run all cells. The CIFAR-10 dataset downloads automatically on the first run.

## Tech Stack
Python, PyTorch, torchvision, NumPy, Matplotlib, Google Colab (Tesla T4 GPU)

## Capstone Repository
[Link to team capstone repo](https://github.com/ORG/REPO-NAME)

## Future Work
- Tune temperature and alpha for better distillation gains
- Train for more epochs and add data augmentation
- Try other compression methods such as pruning and quantization
- Deploy the distilled student on a lightweight web or mobile app
