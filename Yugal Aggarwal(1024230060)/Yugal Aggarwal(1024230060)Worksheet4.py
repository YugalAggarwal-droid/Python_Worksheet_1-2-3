import numpy as np

# -------------------------
# Question 1: Array Creation
# -------------------------
# 1.1
arr1 = np.arange(5, 26)
print("1.1 Array:", arr1)

# 1.2
arr2 = np.random.randint(10, 51, size=(3, 4))
print("\n1.2 2D Random Array:\n", arr2)

# -------------------------
# Question 2: Array Attributes
# -------------------------
print("\n2.1 Attributes of arr1")
print("Shape:", arr1.shape)
print("Size:", arr1.size)
print("Data type:", arr1.dtype)

print("\n2.2 Attributes of arr2")
print("Shape:", arr2.shape)
print("Size:", arr2.size)
print("Data type:", arr2.dtype)

# -------------------------
# Question 3: Array Operations
# -------------------------
Array1 = np.array([2, 4, 6, 8, 10])
Array2 = np.array([1, 3, 5, 7, 9])

print("\n3.2 Array Operations")
print("Addition:", Array1 + Array2)
print("Subtraction:", Array1 - Array2)
print("Multiplication:", Array1 * Array2)
print("Division:", Array1 / Array2)

# -------------------------
# Question 4: Broadcasting
# -------------------------
arr3 = np.arange(1, 10).reshape(3, 3)
print("\n4.1 2D Array:\n", arr3)
print("4.2 After Broadcasting:\n", arr3 * 5)

# -------------------------
# Question 5: Slicing and Indexing
# -------------------------
arr4 = np.arange(10, 26).reshape(4, 4)
print("\n5.1 4x4 Array:\n", arr4)
print("5.2 Second Row:", arr4[1])
print("5.2 Last Column:", arr4[:, -1])
arr4[0] = 0
print("5.3 First row replaced with zeros:\n", arr4)

# -------------------------
# Question 6: Boolean Indexing
# -------------------------
arr5 = np.random.randint(20, 41, size=10)
print("\n6.1 Random Array:", arr5)
print("6.2 Elements > 30:", arr5[arr5 > 30])

# -------------------------
# Question 7: Reshaping
# -------------------------
arr6 = np.arange(11, 23)
reshaped_arr6 = arr6.reshape(3, 4)
print("\n7.1 1D Array:", arr6)
print("7.2 Reshaped to 3x4:\n", reshaped_arr6)

# -------------------------
# Question 8: Matrix Operations
# -------------------------
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("\n8.1 Matrices")
print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("8.2 Matrix Multiplication:\n", A @ B)
print("Transpose of A:\n", A.T)

# -------------------------
# Question 9: Statistical Functions
# -------------------------
arr7 = np.random.randint(10, 61, size=15)
print("\n9.1 Random Array:", arr7)
print("Mean:", np.mean(arr7))
print("Median:", np.median(arr7))
print("Standard Deviation:", np.std(arr7))

# -------------------------
# Question 10: Linear Algebra
# -------------------------
A = np.array([[2, 1, 3],
              [0, 5, 6],
              [7, 8, 9]])
print("\n10.1 Matrix A:\n", A)
print("Determinant:", np.linalg.det(A))
print("Inverse:\n", np.linalg.inv(A))
eigvals, eigvecs = np.linalg.eig(A)
print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)

# -------------------------
# Question 11: Robot Path
# -------------------------
positions = np.array([[0, 0], [2, 3], [4, 7], [7, 10], [10, 15]])
# 11.1 Euclidean distance between first and last
dist_first_last = np.linalg.norm(positions[-1] - positions[0])
print("\n11.1 Euclidean Distance:", dist_first_last)

# 11.2 Total distance step by step
step_dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
total_dist = np.sum(step_dists)
print("11.2 Total Distance:", total_dist)
