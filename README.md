# MNIST Digit Classification

This repository contains an implementation of a neural network for classifying handwritten digits from the MNIST dataset. The model is built using TensorFlow and Keras and demonstrates preprocessing, training, and evaluation workflows.

## Dataset

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels:
- **Training set**: 60,000 images
- **Test set**: 10,000 images

## Model Architecture

The implemented neural network is a feedforward fully connected model with the following layers:

1. **Input Layer**:
   - Flattens the input images (28x28) into a vector of 784 features.
2. **Hidden Layers**:
   - Dense layer with 128 neurons and ReLU activation.
   - Dense layer with 64 neurons and Sigmoid activation.
   - Dense layer with 32 neurons and Sigmoid activation.
3. **Output Layer**:
   - Dense layer with 10 neurons (one for each class) and Softmax activation.

## Preprocessing

- The pixel values of the images are normalized to the range [0, 1].
- The input images are flattened to 1D vectors to match the input requirements of the Dense layers.

## Training

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Cross-Entropy
- **Metrics**: Accuracy
- **Epochs**: 10

## Results

The model was trained for 10 epochs, and the test set was evaluated to determine its performance. The accuracy on the test set is approximately **97%**.

## Error Analysis

The notebook includes an error analysis section where:
- Incorrect predictions are identified.
- The corresponding images, true labels, and predicted labels are visualized for inspection.

## Usage

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- TensorFlow
- Matplotlib
- NumPy

### Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Run the Jupyter notebook:
   ```bash
   jupyter notebook mnist.ipynb
   ```

## Visualizations

The notebook visualizes:
- A sample image from the dataset with its label.
- Incorrect predictions from the test set, highlighting potential areas for model improvement.

## Future Improvements

- Experiment with different activation functions and optimizers.
- Add convolutional layers for better feature extraction.
- Implement additional regularization techniques to further reduce overfitting.

## License

This project is licensed under the MIT License.

---

Feel free to raise issues or submit pull requests for enhancements!

