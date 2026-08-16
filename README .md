# Breast Cancer Classification using Neural Networks

A **binary breast cancer classification project** implemented using **PyTorch** to classify tumors as **Benign (B)** or **Malignant (M)** based on numerical diagnostic features.

The project demonstrates the complete machine learning pipeline — from data preprocessing and feature scaling to building, training, and evaluating a neural network classifier.

## Project Overview

The objective of this project is to predict whether a breast tumor is:

- **Benign (B)** — non-cancerous
- **Malignant (M)** — cancerous

The dataset contains numerical features describing characteristics of cell nuclei, including measurements related to:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

The project contains implementations demonstrating both the underlying neural-network operations and a cleaner PyTorch `nn.Module` based training pipeline.

---

## Tech Stack

- **Python**
- **PyTorch**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Jupyter Notebook / Google Colab**

---

## Repository Structure

```text
Breast_cancer_classification_using_neural_networks/
│
├── Breast_cancer.ipynb
│
├── breast_cancer_classification_pipeline_with__nn_module.ipynb
│
└── README.md
```

### `Breast_cancer.ipynb`

Demonstrates the neural-network workflow with more direct control over operations such as:

- Weight initialization
- Bias initialization
- Forward propagation
- Sigmoid activation
- Binary cross-entropy loss
- Gradient calculation
- Parameter updates

### `breast_cancer_classification_pipeline_with__nn_module.ipynb`

Implements the classification pipeline using PyTorch's standard abstractions:

- `torch.nn.Module`
- `nn.Linear`
- `nn.Sigmoid`
- `nn.BCELoss`
- `torch.optim.SGD`
- `Dataset`
- `DataLoader`

---

## Machine Learning Pipeline

The complete workflow is:

```text
Breast Cancer Dataset
        |
        v
Data Cleaning
        |
        v
Feature / Target Separation
        |
        v
Train-Test Split
        |
        v
Feature Standardization
        |
        v
Label Encoding
        |
        v
NumPy → PyTorch Tensors
        |
        v
Dataset & DataLoader
        |
        v
Neural Network
        |
        v
Forward Propagation
        |
        v
Binary Cross-Entropy Loss
        |
        v
Backpropagation
        |
        v
SGD Parameter Update
        |
        v
Model Evaluation
```

---

## Dataset

The dataset contains **569 observations** and initially contains 33 columns.

Two non-feature columns are removed:

```python
df = df.drop(
    columns=["id", "Unnamed: 32"],
    errors="ignore"
)
```

After preprocessing, the model uses **30 numerical input features** for breast cancer classification.

The target column is:

```text
diagnosis
```

with two classes:

```text
B → Benign
M → Malignant
```

---

## Data Preprocessing

### 1. Train-Test Split

The dataset is divided into training and testing sets using an **80/20 split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:],
    df.iloc[:, 0],
    test_size=0.2
)
```

This produces approximately:

```text
Training samples → 455
Testing samples  → 114
```

---

## Feature Scaling

The numerical features have very different ranges.

For example, measurements such as area can be much larger numerically than features such as smoothness.

The project therefore uses:

```python
StandardScaler()
```

The scaler is fitted only on the training data:

```python
scaler.fit_transform(X_train)
```

and the same transformation is applied to the test data:

```python
scaler.transform(X_test)
```

This prevents information from the test set from leaking into model training.

---

## Label Encoding

The original target values are categorical:

```text
B
M
```

`LabelEncoder` converts these labels into numerical values suitable for model training:

```text
B → 0
M → 1
```

The task therefore becomes a standard **binary classification problem**.

---

## Conversion to PyTorch Tensors

After preprocessing, NumPy arrays are converted into PyTorch tensors:

```python
X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor = torch.from_numpy(X_test).float()

y_train_tensor = torch.from_numpy(y_train).float()
y_test_tensor = torch.from_numpy(y_test).float()
```

The training feature tensor has the shape:

```text
455 × 30
```

where:

```text
455 → training examples
30  → input features
```

---

## Dataset and DataLoader

A custom PyTorch `Dataset` is used to pair each feature vector with its corresponding label.

```python
class CustomDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
```

PyTorch `DataLoader` then creates mini-batches:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
```

Mini-batch training makes the training pipeline more scalable and allows parameters to be updated after processing smaller groups of samples.

---

## Neural Network Architecture

The `nn.Module` implementation uses a simple binary classifier:

```python
class MySimpleNN(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.linear = nn.Linear(
            num_features,
            1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, features):

        out = self.linear(features)
        out = self.sigmoid(out)

        return out
```

Architecture:

```text
30 Input Features
        |
        v
Linear Layer
30 --------> 1
        |
        v
Sigmoid
        |
        v
Probability
0 --------> 1
```

The model produces a probability representing the predicted class.

---

## Forward Propagation

For every sample, the linear layer first calculates:

```text
z = XW + b
```

where:

```text
X → input features
W → trainable weights
b → trainable bias
```

The result is passed through the sigmoid activation function:

```text
             1
σ(z) = ---------------
        1 + e^(-z)
```

This converts the output into a value between:

```text
0 and 1
```

which can be interpreted as a binary classification probability.

---

## Loss Function

The model uses **Binary Cross-Entropy Loss**:

```python
loss_function = nn.BCELoss()
```

Binary Cross-Entropy measures the difference between the predicted probability and the actual binary label.

The training objective is to minimize this loss.

---

## Optimizer

The project uses **Stochastic Gradient Descent (SGD)**:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)
```

The optimizer updates the model's trainable weights and bias based on the gradients calculated during backpropagation.

---

## Training Process

The model is trained for:

```text
25 epochs
```

with:

```text
Batch Size    → 32
Learning Rate → 0.1
Optimizer     → SGD
Loss          → Binary Cross-Entropy
```

The core training loop follows:

```python
for epoch in range(epochs):

    for batch_features, batch_labels in train_loader:

        # Forward propagation
        y_pred = model(batch_features)

        # Calculate loss
        loss = loss_function(
            y_pred,
            batch_labels.view(-1, 1)
        )

        # Clear previous gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()
```

The process can be summarized as:

```text
Forward Pass
     ↓
Prediction
     ↓
Calculate Loss
     ↓
Zero Existing Gradients
     ↓
Backpropagation
     ↓
Calculate Gradients
     ↓
SGD Updates Weights
     ↓
Repeat
```

---

## Model Evaluation

During evaluation, the model is switched to evaluation mode:

```python
model.eval()
```

Gradient calculation is disabled:

```python
with torch.no_grad():
```

The sigmoid output is converted into a binary prediction using a threshold of `0.5`:

```python
y_pred = (y_pred > 0.5).float()
```

Therefore:

```text
Probability ≤ 0.5 → Class 0
Probability > 0.5 → Class 1
```

The recorded notebook run achieves an overall test accuracy of approximately:

```text
96.27%
```

> The exact result may vary between runs because model initialization, data splitting, and batch shuffling can introduce randomness.

---

## Why PyTorch?

PyTorch was used because it provides both:

1. Low-level tensor operations for understanding neural-network fundamentals.
2. High-level abstractions such as `nn.Module`, optimizers, datasets, and data loaders for building structured training pipelines.

This repository demonstrates both approaches, making it useful for understanding what happens internally during neural-network training as well as how PyTorch is typically used to structure ML workflows.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/MARKEE-code-sketch/Breast_cancer_classification_using_neural_networks.git
```

Move into the project directory:

```bash
cd Breast_cancer_classification_using_neural_networks
```

Install the required dependencies:

```bash
pip install numpy pandas torch scikit-learn jupyter
```

---

## Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open either:

```text
Breast_cancer.ipynb
```

or:

```text
breast_cancer_classification_pipeline_with__nn_module.ipynb
```

and execute the cells sequentially.

The notebooks can also be uploaded and executed using **Google Colab**.

---

## Key Concepts Demonstrated

- Binary Classification
- Artificial Neural Networks
- PyTorch Tensors
- Feature Standardization
- Label Encoding
- Train-Test Splitting
- PyTorch `Dataset`
- PyTorch `DataLoader`
- Mini-Batch Training
- Forward Propagation
- Sigmoid Activation
- Binary Cross-Entropy Loss
- Backpropagation
- Gradient Descent
- SGD Optimization
- Model Evaluation

---

## Future Improvements

Possible extensions include:

- Add precision, recall and F1-score evaluation
- Add confusion matrix visualization
- Use stratified train-test splitting
- Compare SGD with Adam
- Add hidden layers to experiment with deeper architectures
- Introduce validation data and early stopping
- Experiment with learning rates and batch sizes
- Save and reload trained model checkpoints
- Add reproducible random seeds
- Build an inference function for new samples

---

## Disclaimer

This repository is an educational machine learning project and is **not intended for clinical diagnosis or medical decision-making**.

---

## Author

**Mrinal Kumar**

GitHub: [MARKEE-code-sketch](https://github.com/MARKEE-code-sketch)

## Repository

[Breast Cancer Classification using Neural Networks](https://github.com/MARKEE-code-sketch/Breast_cancer_classification_using_neural_networks)
