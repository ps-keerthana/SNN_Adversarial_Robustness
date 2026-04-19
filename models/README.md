# Model Weights

Pre-trained model weights saved as PyTorch `.pt` files. All models trained using **snnTorch** with surrogate gradient descent.

| File | Architecture | Dataset | Encoding | Clean Acc |
|---|---|---|---|---|
| `snn_rate_mnist.pt` | LIF-MLP | MNIST | Rate | 97.9% |
| `snn_temporal_mnist.pt` | LIF-MLP | MNIST | Temporal | 97.8% |
| `snn_rate_cifar.pt` | Conv-SNN | CIFAR-10 | Rate | 69.6% |
| `snn_temporal_cifar.pt` | Conv-SNN | CIFAR-10 | Temporal | 79.3% |
| `ann_mnist.pt` | MLP (ReLU) | MNIST | — | 98.4% |
| `cnn_cifar.pt` | VGG-style CNN | CIFAR-10 | — | 89.5% |

> **Note:** `.pt` files are tracked with Git LFS due to their size (3–10 MB each).  
> To download: `git lfs pull`
