# 线性回归: 逆矩阵方法
#----------------------------------
#
# This function shows how to use TensorFlow to
# solve linear regression via the matrix inverse.
#
# Given Ax=b, solving for x:
#  x = (t(A) * A)^(-1) * t(A) * b
#  where t(A) is the transpose of A

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# 初始化计算图
sess = tf.Session()

# 生成数据
#x_vals = np.linspace(0, 10, 100)
#y_vals = x_vals + np.random.normal(0, 1, 100)

x_vals=[48.8, 49, 49.2, 49.4, 49.6, 49.8, 50.0, 50.2, 50.4, 50.6]
y_vals=[90.08, 83.74, 76.63, 69.95,63.34, 56.58, 49.89, 43.01, 36.35, 29.83]
print(len(x_vals))
print(len(y_vals))

# 创建后续求逆方法所需的矩阵。
# 创建A矩阵，其为矩阵x_vals_column和ones_column的合并。
x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 10)))
A = np.column_stack((x_vals_column, ones_column))

# 然后以矩阵y_vals创建b矩阵
b = np.transpose(np.matrix(y_vals))

# 将A和矩阵转换成张量
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

# 逆矩阵方法（Matrix inverse solution）
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
tA_A_inv = tf.matrix_inverse(tA_A)
product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
solution = tf.matmul(product, b_tensor)

solution_eval = sess.run(solution)

# 从解中抽取系数、斜率和y截距y-intercept
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]

print('slope: ' + str(slope))
print('y_intercept: ' + str(y_intercept))

# 求解拟合直线
best_fit = []
for i in x_vals:
  best_fit.append(slope*i+y_intercept)

# 绘制结果
plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.show()