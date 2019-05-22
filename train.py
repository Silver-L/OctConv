'''
# cifar10/cifar100 classification (train)
# Author: Zhihui Lu
# Date: 2019/05/15
'''

import os
import tensorflow as tf
import numpy as np
import utils
from tqdm import tqdm
from absl import flags, app

from resnet50 import normal_resnet50, octconv_resnet50
from model import resnet_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Surpress verbose warnings


# flag
FLAGS = flags.FLAGS
flags.DEFINE_string("outdir", "", "output directory")
flags.DEFINE_string("gpu_index", "0", "GPU-index")
flags.DEFINE_string("problem", "cifar100", "dateset(cifar10 or cifar100)")
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_integer("epoch", 200, "number of epoch")
flags.DEFINE_float("alpha", 0.25, "hyperparameter of octconv")
flags.DEFINE_list("image_size", [32, 32, 3], "image size")
flags.DEFINE_bool("is_octconv", True, "is octconv")

def main(argv):

    # turn off log message
    tf.logging.set_verbosity(tf.logging.WARN)

    # check folder
    if not (os.path.exists(os.path.join(FLAGS.outdir, 'tensorboard'))):
        os.makedirs(os.path.join(FLAGS.outdir, 'tensorboard'))
    if not (os.path.exists(os.path.join(FLAGS.outdir, 'model'))):
        os.makedirs(os.path.join(FLAGS.outdir, 'model'))


    # load train data(cifar10, class: 10)
    if FLAGS.problem == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        num_class = 10
    # load train data(cifar100, class: 100)
    if FLAGS.problem == 'cifar100':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
        num_class = 100

    # preprocess
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255, horizontal_flip=True,
                                    width_shift_range=4.0/32.0, height_shift_range=4.0/32.0)
    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

    # initializer
    init_op = tf.group(tf.initializers.global_variables(),
                       tf.initializers.local_variables())


    with tf.Session(config = utils.config(index=FLAGS.gpu_index)) as sess:

        if FLAGS.is_octconv:
            network = octconv_resnet50
        else:
            network = normal_resnet50

        # set network
        kwargs = {
            'sess': sess,
            'outdir': FLAGS.outdir,
            'input_size': FLAGS.image_size,
            'alpha': FLAGS.alpha,
            'network': network,
            'num_class': num_class,
            'is_training':True,
            'learning_rate': 1e-4
        }

        Model = resnet_model(**kwargs)

        utils.cal_parameter()

        # prepare tensorboard
        writer_train = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'train'), sess.graph)

        value_loss = tf.Variable(0.0)
        tf.summary.scalar("train_loss", value_loss)
        merge_op = tf.summary.merge_all()

        # initialize
        sess.run(init_op)

        tbar = tqdm(range(FLAGS.epoch), ascii=True)
        epoch_loss = []
        for i in tbar:
            train_data_shuffled = train_gen.flow(x_train, y_train, FLAGS.batch_size, shuffle=True)

            # one epoch
            for iter in range(x_train.shape[0]//FLAGS.batch_size):
                train_data_batch = next(train_data_shuffled)

                label = tf.keras.utils.to_categorical(train_data_batch[1], num_classes=num_class)

                # training
                train_loss = Model.update(train_data_batch[0], label)
                epoch_loss.append(np.mean(train_loss))

                s = "epoch:{}, step:{}, Loss: {:.4f}".format(i, iter, np.mean(epoch_loss))
                tbar.set_description(s)

            summary_train_loss = sess.run(merge_op, {value_loss: np.mean(epoch_loss)})
            writer_train.add_summary(summary_train_loss, i)

            epoch_loss.clear()

            # save model
            Model.save_model(i)

if __name__ == '__main__':
    app.run(main)
