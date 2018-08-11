import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01
training_epochs = 100

x_train = np.linspace(-1, 1, 101)
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33

X = tf.placeholder("float")
Y = tf.placeholder("float")
def model(X, w):
    return tf.multiply(X, w)

w = tf.Variable(0.0, name="weights")

y_model = model(X, w)
cost = tf.square(Y - y_model)

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(training_epochs):
    for (x, y) in zip(x_train, y_train):
        sess.run(train_op, feed_dict={X: x, Y: y})

w_val = sess.run(w)

sess.close()
plt.scatter(x_train, y_train)
y_learned = x_train * w_val
plt.plot(x_train, y_learned, 'r')
plt.show()

'''
最小化成本函数值，TensorFlow 试图以有效的方式更新参数，并最终达到最佳的可能值。每个更新所有参数的步骤称为 epoch。
无论哪个参数 w，最优的成本函数值都是最小的。成本函数的定义是真实值与模型响应之间的误差的范数(norm，可以是 2 次方、绝对值、3 次方……)。最后，响应值由模型的函数计算得出。
在本例中，成本函数定义为误差的和(sum of errors)。通常用实际值 f(x) 与预测值 M(w，x) 之间的平方差来计算预测 x 的误差。
该代码定义了成本函数，并要求 TensorFlow 运行(梯度下降)优化来找到最佳的模型参数。s
'''