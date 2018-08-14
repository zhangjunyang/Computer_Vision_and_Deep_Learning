import tensorflow as tf
hello = tf.constant('first tensorflow')
sess = tf.Session()
print(sess.run(hello))