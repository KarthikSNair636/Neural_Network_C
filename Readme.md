# 🧠 MNIST Neural Network in C

A simple **feedforward neural network** implemented in **C** for recognizing handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).  
One hidden layer with 128 neurons.

---

## 📂 Project Structure

NEURAL_NETWORK_C/
└── mnist_nn/
├── data/ # MNIST dataset files
├── nn.c # Neural network implementation
├── nn.h # Neural network header
├── mnist_loader.c # MNIST dataset loader
├── mnist_loader.h # Loader header
├── train.c # Training routine
├── test.c # Testing routine
└── model.bin # Saved trained model weights


---

## 🏗️ Architecture

| Layer    | Neurons | Activation |
|----------|---------|------------|
| Input    | 784     | –          |
| Hidden   | 128     | ReLU       |
| Output   | 10      | Softmax    |

**Total parameters:** 101,280  
- Input → Hidden: 784 × 128 = 100,352  
- Hidden → Output: 128 × 10 = 1,280  

---

## ⚙️ Training

- **Epochs:** 5  
- **Learning rate:** 0.1  
- **Optimizer:** Gradient Descent  
- **Loss:** Cross-entropy  

> Achieved ~9.8% accuracy after 5 epochs  

---

## 🚀 Usage

### Prerequisites
- C compiler (e.g., `gcc`)  
- MNIST dataset files in `data/`:  
  - [`train-images-idx3-ubyte`](http://yann.lecun.com/exdb/mnist/)  
  - [`train-labels-idx1-ubyte`](http://yann.lecun.com/exdb/mnist/)  
  - [`t10k-images-idx3-ubyte`](http://yann.lecun.com/exdb/mnist/)  
  - [`t10k-labels-idx1-ubyte`](http://yann.lecun.com/exdb/mnist/)  

### Compile
```bash
gcc -o train train.c nn.c mnist_loader.c -lm
gcc -o test test.c nn.c mnist_loader.c -lm

./train
./test

📝 Notes

Single-threaded, uses arrays.

Accuracy limited due to few epochs and simple training.

Improvements: more epochs, mini-batch training, better initialization, more layers.

📚 References

MNIST Dataset

Standard feedforward and backpropagation algorithms
