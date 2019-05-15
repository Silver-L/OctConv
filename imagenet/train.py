import os
import tensorflow as tf
import numpy as np
import random
import glob
import utils
from tqdm import tqdm
from absl import flags, app

from resnet50 import normal_resnet50, octconv_resnet50
from model import resnet_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Surpress verbose warnings


# flag
FLAGS = flags.FLAGS
flags.DEFINE_string("indir", "H:/data_set/tfrecord/imagenet/train", "tfrecord(train) directory")
flags.DEFINE_string("outdir", "H:/experiment_result/octconv/normal_resnet50", "output directory")
flags.DEFINE_string("gpu_index", "0", "GPU-index")
flags.DEFINE_integer("shuffle_buffer_size", 1024, "buffer size of shuffle")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("num_data", 1281167, "number of total data")
flags.DEFINE_integer("epoch", 200, "number of epoch")
flags.DEFINE_float("alpha", 0.125, "hyperparameter of octconv")
flags.DEFINE_list("image_size", [224, 224, 3], "image size")
flags.DEFINE_bool("is_octconv", True, "is octconv")

def main(argv):
    # check folder
    if not (os.path.exists(os.path.join(FLAGS.outdir, 'tensorboard'))):
        os.makedirs(os.path.join(FLAGS.outdir, 'tensorboard'))
    if not (os.path.exists(os.path.join(FLAGS.outdir, 'model'))):
        os.makedirs(os.path.join(FLAGS.outdir, 'model'))

    # get file list
    train_data_list = glob.glob(FLAGS.indir + '/*')
    # shuffle list
    random.shuffle(train_data_list)

    # load train data
    train_set = tf.data.Dataset.list_files(train_data_list)
    train_set = train_set.apply(
        tf.contrib.data.parallel_interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length = os.cpu_count()))
    train_set = train_set.map(utils._parse_function, num_parallel_calls=os.cpu_count())
    train_set = train_set.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    train_set = train_set.repeat()
    train_set = train_set.batch(FLAGS.batch_size)
    train_set = train_set.prefetch(1)
    train_iter = train_set.make_one_shot_iterator()
    train_data = train_iter.get_next()

    # step of each epoch
    if FLAGS.num_data % FLAGS.batch_size == 0:
        step_of_epoch = FLAGS.num_data / FLAGS.batch_size
    else:
        step_of_epoch = FLAGS.num_data // FLAGS.batch_size + 1

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

            # one epoch
            for step in range(step_of_epoch):
                train_data_batch, train_label_batch = sess.run(train_data)
                train_loss = Model.update(train_data_batch, train_label_batch)
                epoch_loss.append(train_loss)
                s = "epoch:{}, step:{}, Loss: {:.4f}".format(i, step, np.mean(epoch_loss))
                tbar.set_description(s)

            summary_train_loss = sess.run(merge_op, {value_loss: np.mean(epoch_loss)})
            writer_train.add_summary(summary_train_loss, i)

            epoch_loss.clear()

            # save model
            Model.save_model(i)

if __name__ == '__main__':
    app.run(main)
