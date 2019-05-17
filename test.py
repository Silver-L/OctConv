'''
# cifar100 classification (test)
# Author: Zhihui Lu
# Date: 2019/05/17
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
flags.DEFINE_string("model_path", "H:/experiment_result/octconv/octconv_resnet50_0.125/model", "model folder path")
flags.DEFINE_string("outdir", "H:/experiment_result/octconv/octconv_resnet50_0.125", "output directory")
flags.DEFINE_string("gpu_index", "0", "GPU-index")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("epoch", 200, "number of epoch")
flags.DEFINE_float("alpha", 0.125, "hyperparameter of octconv")
flags.DEFINE_list("image_size", [32, 32, 3], "image size")
flags.DEFINE_bool("is_octconv", True, "is octconv")

def main(argv):

    # turn off log message
    tf.logging.set_verbosity(tf.logging.WARN)

    # check folder
    if not (os.path.exists(os.path.join(FLAGS.outdir, 'tensorboard'))):
        os.makedirs(os.path.join(FLAGS.outdir, 'tensorboard'))

    # load train data(cifar100, class: 100)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

    # preprocess
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
            'is_training':False,
            'learning_rate': 1e-4
        }

        Model = resnet_model(**kwargs)

        utils.cal_parameter()

        # prepare tensorboard
        writer_test = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'test'))

        value_acc = tf.Variable(0.0)
        tf.summary.scalar("test_accuracy", value_acc)
        merge_op = tf.summary.merge_all()

        # initialize
        sess.run(init_op)

        tbar = tqdm(range(FLAGS.epoch), ascii=True)
        epoch_acc = []
        for i in tbar:
            test_data = test_gen.flow(x_train, y_train, FLAGS.batch_size, shuffle=False)

            # one epoch
            Model.restore_model(FLAGS.model_path + '/model_{}'.format(i))
            for iter in range(x_test.shape[0]//FLAGS.batch_size):
                train_data_batch = next(test_data)

                label = tf.keras.utils.to_categorical(train_data_batch[1], num_classes=100)

                test_acc = Model.test(train_data_batch[0], label)
                epoch_acc.append(np.mean(test_acc))

                s = "epoch:{}, acc: {:.4f}".format(i, np.mean(epoch_acc))
                tbar.set_description(s)

            summary_test_acc = sess.run(merge_op, {value_acc: np.mean(epoch_acc)})
            writer_test.add_summary(summary_test_acc, i)

            epoch_acc.clear()


if __name__ == '__main__':
    app.run(main)
