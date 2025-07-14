# Neural Network from scratch for Computer Vision

A suite of MNIST digit‑recognition projects showcasing:
1. A from‑scratch MLP in pure NumPy  
2. A custom SGD training/optimization loop in TensorFlow/Keras  
3. Various autoencoder architectures (linear, deep, denoising) in Keras

---

## 📂 Repository Structure

```bash
.
├── README.md
├── RayaNN/                        # Pure‑Python scratch MLP implementation
    ├── save_model.npz                 # Weights from the scratch MLP after training
│   ├── rayane_nn.py                   # Training script
│   └── main.py                        # Used to test the model visually    
├── Autoencoder_NN_DAKHLAOUI.ipynb # Keras autoencoder lab: PCA comparison, deep & denoising AEs
└── Optimization_MNIST_Dakhlaoui.ipynb  # Custom training loop lab: gradient‑tape, SGD, LR experiments

                                   


```

## 🧑‍💻 Notebooks & Code

## 1️⃣ MLP From Scratch (`RayaNN/`)  
**Goal**: Implement a 2-layer neural network from scratch (NumPy only) for MNIST classification.  

**Key Features**:  
- **Custom Layers**:  
  - `MLP` (Linear) with forward/backward passes  
  - `ReLU` activation  
  - `LogSoftmax` + `NLLLoss` (numerically stable)  
- **Training Core**:  
  - Batch gradient descent (`batch_size=100`)  
  - Manual backpropagation chain  
  - Learning rate scheduling (`lr=1e-3`)  
- **Results**:  
  - Model saving/loading via `.npz`  
  - ~93% dev accuracy  



## 2️⃣ Custom Optimization Loop (`Optimization_MNIST_Dakhlaoui.ipynb/`)  

**Goal**: Implement manual SGD with `tf.GradientTape` on MNIST and analyze hyperparameters.  

**Key Points**:  
- Gradient computation via `GradientTape`  
- Custom weight updates (`update_weights()` function)  
- Loss/accuracy tracking (train/test sets)  
- Experiments: Learning rates (0.1, 0.01, 0.001), batch sizes, epochs  
- Results: ~90% test accuracy, convergence analysis  

**Skills**: TensorFlow autodiff, optimization debugging, hyperparameter tuning.  



## 3️⃣ Autoencoders (`Autoencoder_NN_Dakhlaoui.ipynb/`)  

**Goal**: Compare PCA with neural autoencoders and implement denoising on MNIST.  

**Key Points**:  
- **PCA vs Linear AE**:  
  - 2D latent space visualization  
  - Variance comparison (PCA: ~16.8% vs AE reconstruction)  
- **Deep Autoencoder**:  
  - Non-linear architecture (512 → 256 → 128 units)  
  - Improved reconstruction quality  
- **Denoising AE**:  
  - Noise injection (factor=0.4)  
  - Multi-layer encoder/decoder with ReLU  
- **Metrics**: MSE loss, visual reconstruction analysis   
