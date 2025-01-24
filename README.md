# Linear Regression from Scratch and PyTorch Implementation

## Overview

This repository demonstrates two implementations of linear regression:

1. **Vanilla and Regularized Linear Regression from Scratch (using NumPy)**
   - Implements linear regression with Mean Squared Error (MSE) as the cost function.
   - Adds L2 regularization to control overfitting by penalizing large coefficients.
   - Explores the effect of varying the regularization parameter `λ` on the regression line.

2. **Polynomial Regression using PyTorch**
   - Fits a degree-5 polynomial regression model on the `curve80.txt` dataset.
   - Utilizes PyTorch to perform gradient descent for optimization.

## Key Components

### Linear Regression from Scratch (NumPy)

- **Data Generation**: 
  - Simulated using `y = x - 3 + ε`, where `ε ~ N(0, 0.3)`.

- **Cost Function**:
  - Without Regularization:
    ```
    J(W) = (1 / 2m) * Σ(h(x_i) - y_i)^2
    ```
  - With L2 Regularization:
    ```
    J(W) = (1 / 2m) * Σ(h(x_i) - y_i)^2 + (λ / 2m) * ||W||^2
    ```

- **Gradient Descent**:
  - Regularized gradient:
    ```
    ∇J(W) = (1 / m) * X^T * (XW - y) + (λ / m) * W
    ```
  - Iteratively updates weights using:
    ```
    W = W - α * ∇J(W)
    ```

- **Effect of Regularization**:
  - Models trained with varying `λ` values show increasing regularization flattens the regression line.
  
### Polynomial Regression with PyTorch

- **Dataset**: 
  - `curve80.txt`, split into 75% training and 25% testing data.
  
- **Feature Transformation**:
  - Generates degree-5 polynomial features.
  - Rescales features for numerical stability.

- **PyTorch Implementation**:
  - Initializes a linear regression model with 5 features and 1 output.
  - Uses Mean Squared Error (MSE) loss and Stochastic Gradient Descent (SGD) optimizer.
  - Trains the model over 100,000 epochs, tracking the loss.

- **Visualization**:
  - Plots training loss vs. epochs to show convergence.
  - Visualizes the prediction function vs. the training data.

## Results

1. **From-Scratch Regularized Regression**:
   - As `λ` increases, the model prioritizes minimizing coefficient magnitudes, reducing overfitting but at the cost of flexibility.

2. **PyTorch Polynomial Regression**:
   - The trained model successfully fits the non-linear curve, demonstrating PyTorch’s utility for more complex regression tasks.

