'''
@ TFRecord generator (dataset: imagenet)
@ Author: Zhihui Lu
@ Date: 2019/05/12
@ Reference: https://blog.csdn.net/gzroy/article/details/83416339
'''

import os
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
from absl import flags, app

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Surpress verbose warnings


# flag
FLAGS = flags.FLAGS
flags.DEFINE_string("data_indir", "/mnt/user/data/imagenet/image/val", "data directory")
flags.DEFINE_string("data_list_path", "H:/data_set/image/imagenet/validation_label.txt", "data list path (.txt)")
flags.DEFINE_string("outdir", "H:/data_set/tfrecord/imagenet/tiger/val", "output directory")
flags.DEFINE_integer("num_per_tfrecord", 1000, "number per tfrecord")
flags.DEFINE_bool("is_shuffle", True, "shuffle or not")

def main(argv):
    #check folder
    if not (os.path.exists(FLAGS.outdir)):
        os.makedirs(FLAGS.outdir)

    # load info list
    info_list = np.loadtxt(FLAGS.data_list_path, dtype=str, delimiter=" ")

    # shuffle
    if FLAGS.is_shuffle == True:
        np.random.shuffle(info_list)

    # data list
    data_list = []
    label_list = []
    for i in range(info_list.shape[0]):
        data_list.append(FLAGS.data_indir + "/" + info_list[i][0])
        label_list.append(info_list[i][1])


    num_per_tfrecord = int(FLAGS.num_per_tfrecord)
    num_of_total_image = len(data_list)

    if (num_of_total_image % num_per_tfrecord != 0):
        num_of_recordfile = num_of_total_image // num_per_tfrecord + 1
    else:
        num_of_recordfile = num_of_total_image // num_per_tfrecord

    num_per_tfrecord_final = num_of_total_image - num_per_tfrecord * (num_of_recordfile - 1)

    print('number of total TFrecordfile: {}'.format(num_of_recordfile))

    # write TFRecord
    for i in range(num_of_recordfile):
        tfrecord_filename = os.path.join(FLAGS.outdir, 'recordfile_{}'.format(i + 1))
        write = tf.python_io.TFRecordWriter(tfrecord_filename)

        print('Writing recordfile_{}'.format(i+1))

        if i == num_of_recordfile - 1:
            loop_buf = num_per_tfrecord_final
        else :
            loop_buf = num_per_tfrecord

        for image_index in range(loop_buf):
            image_path = data_list[image_index + i*num_per_tfrecord]
            label = label_list[image_index + i*num_per_tfrecord]

            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'image_path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path.encode('utf-8')])),
                    'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)]))
                }))

            write.write(example.SerializeToString())
        write.close()


def tensor_to_array(tensor1):
    return tensor1.numpy()


if __name__ == '__main__':
    app.run(main)
