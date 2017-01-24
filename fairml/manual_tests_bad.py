import os

from black_box_functionality import verify_black_box_estimator
from sklearn import linear_model


reg = linear_model.Lasso(alpha = 0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])

value = verify_black_box_estimator(reg, 2)
print(value)