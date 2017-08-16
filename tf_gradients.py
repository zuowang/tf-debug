import tensorflow as tf

# define symbolic variables
x = tf.placeholder("float") 
y = tf.placeholder("float")

# define a function R=R(x,y)
R = 0.127-(x*0.194/(y+0.194))

# The derivative of R with respect to y
Rdy = tf.gradients(R, y); 

# Launch a session for the default graph to comput dR/dy at (x,y)=(0.362, 0.556)
sess = tf.Session()
result = sess.run(Rdy, {x:0.362,y:0.556})
print result
#[0.12484978]
