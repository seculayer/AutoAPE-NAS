from absl import app, flags, logging
from absl.flags import FLAGS
import os
import time
import numpy as np
import tensorflow as tf

from modules.models import CifarModel
from modules.dataset import load_cifar10_dataset
from modules.utils import (
    set_memory_growth, load_yaml, count_parameters_in_MB, ProgressBar,
    AvgrageMeter, accuracy)


flags.DEFINE_string('cfg_path', '../configs/pcdarts_cifar.yaml',
                    'config file path')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = CifarModel(cfg, training=False)
    model.summary(line_length=80)
    print("param size = {:f}MB".format(count_parameters_in_MB(model)))

    # load dataset
    test_dataset = load_cifar10_dataset(
        cfg['test_batch_size'], split='test', shuffle=False,
        drop_remainder=False, using_crop=False, using_flip=False,
        using_cutout=False)

    # load checkpoint
    checkpoint_path = './checkpoints/' + cfg['sub_name'] + '/best.ckpt'
    try:
        model.load_weights('./checkpoints/' + cfg['sub_name'] + '/best.ckpt')
        print("[*] load ckpt from {}.".format(checkpoint_path))
    except:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_path))
        exit()

    # inference
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    start_time = time.time()
    print("Start")

    for step, (inputs, labels) in enumerate(test_dataset):
        # run model
        logits = model(inputs)

        # caclulate top1, top5 acc
        prec1, prec5 = accuracy(logits.numpy(), labels.numpy(), topk=(1, 5))

        n = inputs.shape[0]
        top1.update(prec1, n)
        top5.update(prec5, n)

        print(" {:03d}: top1 {:f}, top5 {:f}".format(step, top1.avg, top5.avg))

    print("End")
    print("Time: {:.4f}sec".format((time.time() - start_time)))

    print("Test Acc: top1 {:.2f}%, top5 {:.2f}%".format(top1.avg, top5.avg))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
