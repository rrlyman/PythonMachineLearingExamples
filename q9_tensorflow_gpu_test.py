# Creates a graph.
import tensorflow as tf
#from tensorflow.compat import v1 as tf

#sess = tf.InteractiveSession()  
@tf.function
def d(a,b):
    return tf.matmul(a, b)

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.

# Runs the op.

# tens1 = tf.constant([ [[1,2],[2,3]], [[3,4],[5,6]] ]) 
# print (sess.run(tens1)[1,1,0])
# self._sess.run(tf.initialize_all_variables())
for i in range(100000):
    d(a,b)
print ('\n########################### No Errors ####################################')