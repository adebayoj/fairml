import numpy as np

a = np.random.uniform(0, 1, 1000)
b = np.random.uniform(0, 1, 1000)
c = np.random.uniform(0, 1, 1000)

A = np.zeros((np.shape(a)[0], 2))
A[:, 0] = a
A[:, 1] = b

a_t_a = np.dot(A.T, A)
partial = np.dot(A, np.linalg.inv(a_t_a))

projection_matrix = np.dot(partial, A.T)
print(projection_matrix.shape)

proj_c = np.dot(projection_matrix, c)

print(np.dot(proj_c, A[:, 0]))
print(np.dot(c - proj_c, A[:, 0]))
print()
print(np.dot(proj_c, A[:, 1]))
print(np.dot(c - proj_c, A[:, 1]))
