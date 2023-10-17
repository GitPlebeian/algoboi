
import tensorflow as tf

print(tf.__version__)

physical_devices = tf.config.list_physical_devices('GPU')
tf.print(physical_devices)

# from tensorflow.python.compiler.mlcompute import mlcompute

# # Checking if TensorFlow is using Apple's Metal
# print(mlcompute.is_apple_mlc_enabled())

# print("Bob Is Dumb")