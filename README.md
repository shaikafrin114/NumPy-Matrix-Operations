# NumPy-Matrix-Operations

## Task Description
Create two matrices and perform addition, subtraction, matrix multiplication, and transpose operations.

## Objective
Understand basic matrix operations used in machine learning.

## Tools Used
- Python
- NumPy
- Jupyter Notebook

## Implementation Code

```python
import numpy as np

# 1. Initialize two 2x2 matrices with sample data
matrix_A = np.array([[4, 7], [2, 6]])
matrix_B = np.array([[1, 3], [5, 8]])

print("Matrix A:\n", matrix_A)
print("\nMatrix B:\n", matrix_B)

# 2. Matrix Addition
addition_result = matrix_A + matrix_B
print("\nAddition Result:\n", addition_result)

# 3. Matrix Subtraction
subtraction_result = matrix_A - matrix_B
print("\nSubtraction Result:\n", subtraction_result)

# 4. Matrix Multiplication (Dot Product)
multiplication_result = matrix_A @ matrix_B
print("\nMatrix Multiplication Result (@):\n", multiplication_result)

# 5. Transpose Operations
transpose_A = matrix_A.T
print("\nTranspose of Matrix A:\n", transpose_A)
```

## Short Explanation of Matrix Multiplication
Matrix multiplication (dot product) calculates the combined effect of rows from the first matrix and columns from the second matrix. 
- **Rule:** The operation is only valid when the number of columns in the first matrix equals the number of rows in the second matrix.
- **Calculation:** The element at position `(i, j)` in the resulting matrix is computed by multiplying the elements of the `i`-th row of the first matrix by the corresponding elements of the `j`-th column of the second matrix, and summing the results.

## Interview Questions
### What is a matrix?
A **matrix** is a rectangular grid or two-dimensional array of numbers organized in horizontal rows and vertical columns. In machine learning, matrices are highly critical tools used to represent datasets, transform data features, and efficiently compute millions of weights simultaneously during model training.
