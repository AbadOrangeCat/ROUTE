import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("TensorFlow built with CUDA:", tf.test.is_built_with_cuda())
print("TensorFlow built with GPU support:", tf.test.is_built_with_gpu_support())
