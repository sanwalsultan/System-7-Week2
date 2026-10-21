# Day 9: Statistics for Data Science

## Topics Covered

- Mean
- Median
- Mode
- Variance
- Standard Deviation

## 1. Mean

Mean is the average of all values in a dataset.

**Formula:**

Mean = Sum of all values / Number of values

**Example:**

Dataset: `[10, 20, 20, 30, 40]`

Mean = 120 / 5 = **24**

```python
import numpy as np

data = [10, 20, 20, 30, 40]

print("Mean:", np.mean(data))
```

## 2. Median

Median is the middle value when the dataset is arranged in ascending order.

**Example:**

Dataset: `[10, 20, 20, 30, 40]`

Median = **20**

For an even number of values, the median is the average of the two middle values.

```python
print("Median:", np.median(data))
```

## 3. Mode

Mode is the value that appears most frequently in a dataset.

**Example:**

Dataset: `[10, 20, 20, 30, 40]`

Mode = **20**

```python
from statistics import mode

print("Mode:", mode(data))
```

## 4. Variance

Variance measures how far the values are spread out from the mean. It is calculated by taking the average of the squared differences between each value and the mean.

**Example:**

Dataset: `[10, 20, 20, 30, 40]`

Population Variance = **104**

```python
print("Population Variance:", np.var(data))
```

Note: `np.var()` calculates population variance by default. Use `np.var(data, ddof=1)` for sample variance.

## 5. Standard Deviation

Standard deviation measures the spread of data around the mean. It is the square root of the variance and uses the same units as the original data.

**Example:**

Dataset: `[10, 20, 20, 30, 40]`

Standard Deviation = √104 ≈ **10.20**

```python
print("Standard Deviation:", round(np.std(data), 2))
```

## Complete Python Program

```python
import numpy as np
from statistics import mode

data = [10, 20, 20, 30, 40]

print("Dataset:", data)
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Mode:", mode(data))
print("Variance:", np.var(data))
print("Standard Deviation:", round(np.std(data), 2))
```

### Expected Output

```text
Dataset: [10, 20, 20, 30, 40]
Mean: 24.0
Median: 20.0
Mode: 20
Variance: 104.0
Standard Deviation: 10.2
```

## Summary

| Concept | Purpose | Result |
|---|---|---:|
| Mean | Calculates the average | 24 |
| Median | Finds the middle value | 20 |
| Mode | Finds the most frequent value | 20 |
| Variance | Measures squared spread | 104 |
| Standard Deviation | Measures spread in original units | 10.20 |


## Tools and Libraries

- Python
- NumPy
- `statistics` module
