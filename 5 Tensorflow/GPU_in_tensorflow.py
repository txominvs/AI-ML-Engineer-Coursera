import tensorflow as tf

tf.debugging.set_log_device_placement(True) # will PRINT which device each tensor or operation belongs to

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs found!")

    for gpu in gpus:
        print("GPU name:", gpu.name)
        tf.config.experimental.set_memory_growth(gpu, True) # only allocate memory dinamically, as much as needed

with tf.device('/CPU:0'): # run on the CPU
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

with tf.device(gpus[0].name): # run on the GPU (default, if available) name='/device:GPU:0'
    c = tf.matmul(a, b)