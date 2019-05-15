import tensorflow as tf
import numpy as np

# session config
def config(index = "0"):
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=index , # specify GPU number
            allow_growth=True
        )
    )
    return config

# calculate total parameters
def cal_parameter():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    return print('Total params: %d ' % total_parameters)

# calculate jaccard
def jaccard(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.double(np.bitwise_and(im1, im2).sum()) / np.double(np.bitwise_or(im1, im2).sum())


# data generator
def batch_iter(data, labels, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X = shuffled_data[start_index: end_index]
                y = shuffled_labels[start_index: end_index]
                yield X, y

    return num_batches_per_epoch, data_generator()


# imagenet parse_function
def _parse_function(record):
    features = tf.parse_single_example(record, features={
        'image_path': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([1], tf.int64)
    })
    filename, label = features['image_path'], features['label']
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_height = tf.shape(image_decoded)[0]
    image_width = tf.shape(image_decoded)[1]
    random_s = tf.random_uniform([1], minval=256, maxval=481, dtype=tf.int32)[0]
    resized_height, resized_width = tf.cond(image_height<image_width,
                lambda: (random_s, tf.cast(tf.multiply(tf.cast(image_width, tf.float64),tf.divide(random_s,image_height)), tf.int32)),
                lambda: (tf.cast(tf.multiply(tf.cast(image_height, tf.float64),tf.divide(random_s,image_width)), tf.int32), random_s))
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize_images(image_float, [resized_height, resized_width])
    image_flipped = tf.image.random_flip_left_right(image_resized)
    image_cropped = tf.random_crop(image_flipped, [224, 224, 3])
    image_distorted = tf.image.random_brightness(image_cropped, max_delta=63)
    image_distorted = tf.image.random_contrast(image_distorted, lower=0.2, upper=1.8)
    image_distorted = tf.image.per_image_standardization(image_distorted)   # Subtract off the mean and divide by the variance of the pixels.
    # image_distorted = tf.transpose(image_distorted, perm=[2, 0, 1])

    # print(tf.shape(image_distorted))

    label = tf.cast(tf.one_hot(label, 1000, 1, 0), 'float32')
    return image_distorted, label


'''
# under eagar mode (conver tensor to numpy)

tf.enable_eager_execution()

def tensor_to_array(tensor1):
    return tensor1.numpy()

'''