import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

class CnnParams:
    def __init__(self, *params):
        self.kernel_size = params[0]
        self.filters = params[1]
        self.stride = params[2]
        self.padding = params[3]
        self.activation = params[4]

class DenseParams:
    def __init__(self, *params):
        self.units = params[0]
        self.activation = params[1]
        self.dropout = params[2]

class Data:
    def __init__(self, path, train_size=0.75, batch_size=16):
        self.path = path
        self.batch_size = batch_size
        self.train_size = train_size

        self.steering = pd.read_csv(os.path.join(path, "labels.csv"), header=None)

        # randomize csv file
        self.steering = self.steering.sample(frac=1).reset_index(drop=True)

        self.num_points = self.steering.shape[0]
        self.n_batches = int(np.ceil(self.num_points/self.batch_size))

    def next_batch(self, training=True):
        '''
            Yields a batch (x, y) of size `batch_size`
        '''

        ind = int(self.steering.shape[0] * self.train_size)
        if training:
            steering = self.steering[[0, 1]][: ind]
        else:
            steering = self.steering[[0, 1]][ind: ]
        steering = steering.reset_index(drop=True)

        num_points = steering.shape[0]
        n_batches = int(np.ceil(num_points/self.batch_size))
        
        for b in range(n_batches):
            # Get start index of batch.
            start = b * self.batch_size
            
            # Get end index of batch.
            end = min(num_points, start+self.batch_size)

            # Read images from start: end
            steering_batch = np.expand_dims(np.asarray(steering[1][start: end]), -1)
            batch_files = steering[0][start: end].tolist()

            batch_imgs = [np.array(Image.open(f)) for f in batch_files]
            batch_imgs = [img[60: 140, :, :] for img in batch_imgs]
            batch_imgs = [np.expand_dims(img, axis=0) for img in batch_imgs]
            
            # yielding will return a generator.
            yield np.concatenate(batch_imgs), steering_batch*90.0

