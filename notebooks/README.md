# Notebooks

## Main Notebook

**`snn_adversarial_attacks.ipynb`** — The final, clean, structured notebook containing everything:
- SNN training (Rate & Temporal coding) on MNIST and CIFAR-10
- ANN (MLP) and CNN (VGG-style) baseline training
- FGSM and PGD adversarial attack evaluation
- All plots and result tables

Run this on Kaggle with GPU for best results (P100/T4 recommended).

## Exploratory Notebooks

Early-stage experiments, kept for reference. These were the individual pieces before the project was unified:

| Notebook | Description |
|---|---|
| `snn_mnist.ipynb` | Initial SNN training on MNIST |
| `snn_cifar10_initial.ipynb` | Initial SNN training on CIFAR-10 |
| `snn_mnist_adversarial_v1.ipynb` | First adversarial attack attempt on SNN-MNIST |
| `snn_mnist_adversarial_comparision.ipynb` | Comparing SNN vs ANN robustness on MNIST |
| `snn-cifar10-adversarial.ipynb` | SNN adversarial attacks on CIFAR-10 |
| `ann-mnist-adversarial.ipynb` | ANN adversarial attacks on MNIST |
| `ann-cifar10-adversarial.ipynb` | ANN adversarial attacks on CIFAR-10 |
| `cnn-mnist-adversarial.ipynb` | CNN adversarial attacks on MNIST |
| `cnn-cifar10-adversarial.ipynb` | CNN adversarial attacks on CIFAR-10 |
