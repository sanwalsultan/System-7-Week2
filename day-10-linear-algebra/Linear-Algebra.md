# Day 10: Linear Algebra Basics

## Topics Covered

- Vectors
- Matrices
- Dot Products
- Basic Linear Algebra Operations using NumPy

## 1. Vectors

A **vector** is a collection of numbers arranged in a single dimension. Vectors are used in data science to represent features, observations, and numerical data.

**Example:**

```python
import numpy as np

vector = np.array([2, 4, 6])

print("Vector:", vector)
print("First element:", vector[0])
print("Vector + 2:", vector + 2)
```

**Output:**

```text
Vector: [2 4 6]
First element: 2
Vector + 2: [4 6 8]
```

### Vector Addition

Two vectors of the same shape can be added element by element.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Addition:", a + b)
```

Output: `[5 7 9]`

## 2. Matrices

A **matrix** is a rectangular arrangement of numbers in rows and columns. Matrices are used to represent datasets, images, and model parameters.

**Example:**

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("Matrix:\n", matrix)
print("Shape:", matrix.shape)
print("First row:", matrix[0])
print("Element at row 1, column 2:", matrix[0, 1])
```

**Output:**

```text
Matrix:
 [[1 2 3]
  [4 5 6]]
Shape: (2, 3)
First row: [1 2 3]
Element at row 1, column 2: 2
```

### Matrix Addition

Matrices with the same shape can be added element by element.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A + B)
```

Output:

```text
[[ 6  8]
 [10 12]]
```

## 3. Dot Product

The **dot product** multiplies corresponding elements of two vectors and adds the results together.

For two vectors:

`a = [1, 2, 3]`

`b = [4, 5, 6]`

Calculation:

Dot product = (1 × 4) + (2 × 5) + (3 × 6)

Dot product = 4 + 10 + 18 = **32**

### Python Example

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = np.dot(a, b)

print("Dot Product:", result)
```

Output:

```text
Dot Product: 32
```

The `@` operator can also calculate the dot product of two vectors.

```python
print(a @ b)
```

Output: `32`

## 4. Matrix Multiplication

Matrix multiplication combines rows from the first matrix with columns from the second matrix. The number of columns in the first matrix must equal the number of rows in the second matrix.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix Multiplication:\n", A @ B)
```

Output:

```text
[[19 22]
 [43 50]]
```

**Important:** `A * B` performs element-wise multiplication, while `A @ B` performs matrix multiplication.

## 5. Complete Python Program

```python
import numpy as np

# 1. Vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Vector A:", a)
print("Vector B:", b)
print("Vector Addition:", a + b)
print("Dot Product:", np.dot(a, b))

# 2. Matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix Addition:\n", A + B)
print("Matrix Multiplication:\n", A @ B)
print("Matrix Shape:", A.shape)
```

## Summary

| Concept | Description |
|---|---|
| Vector | A one-dimensional collection of numbers |
| Matrix | A two-dimensional arrangement of numbers |
| Vector Addition | Adds corresponding vector elements |
| Dot Product | Multiplies corresponding elements and sums them |
| Matrix Addition | Adds corresponding matrix elements |
| Matrix Multiplication | Multiplies rows by columns |
| Shape | Returns the dimensions of an array |


## Tools and Libraries

- Python
- NumPy
