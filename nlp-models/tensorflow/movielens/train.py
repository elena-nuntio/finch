from __future__ import print_function
from model import model_fn
from data import DataLoader

import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.INFO)

NUM_EPOCHS = 50
BATCH_SIZE = 256

def main():
    model = tf.estimator.Estimator(model_fn, params={'lr': 1e-4})
    dl = DataLoader(BATCH_SIZE)
    for i in xrange(NUM_EPOCHS):
        model.train(dl.train_pipeline())
        print('Testing loss:', model.evaluate(dl.eval_pipeline())['mse'])

if __name__ == '__main__':
    main()
