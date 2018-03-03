import numpy as np
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
    def __init__(self, x, y, batch_size=16):
        self.x = x
        self.y = y
        self.batch_size = batch_size

        self.augment_data()
        self._randomize()

        self.num_points = self.x.shape[0]

    def next_batch(self):
        '''
            Yields a batch (x, y) of size `batch_size`
        '''
        
        # Calculate number of batches.
        n_batches = int(np.ceil(self.num_points/self.batch_size))
        
        for b in range(n_batches):
            # Get start index of batch.
            start = b * self.batch_size
            
            # Get end index of batch.
            end = start + self.batch_size
            # Set limit to the ending index.
            end = min(end, self.num_points)
            
            # yielding will return a generator.
            yield (self.X[start: end], self.Y[start: end])

    def augment_data(self):
        print("Performing data augmentation...")
        
        x_copy = np.copy(self.x)
        y_copy = np.copy(self.y)

        self._augment_flip()
        self._augment_brightness()
        
        print("Done with augmentation!")

    def _augment_flip(self):
        '''
        Augmentation function: 
            Horizantally flips the images in dataset and negate the corresponding steering values.
        '''
        x_copy = np.copy(self.x)
        y_copy = np.copy(self.y)

        with tf.Session() as sess:
            x_copy = sess.run(tf.map_fn(tf.image.flip_left_right, x_copy))
        y_copy = -1.0 * y_copy

        self.x = np.vstack((self.x, x_copy))
        self.y = np.vstack((self.y, y_copy))

    def _augment_brightness(self):
        '''
        Augmentation function: 
            Randomly changes brightness of data. Steering value remains same.

        Courtesy: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
        '''
        
        x_copy = np.copy(self.x)
        y_copy = np.copy(self.y)

        with tf.Session() as sess:
            x_copy = sess.run(tf.image.rgb_to_hsv(x_copy))

            random_bright = 0.5+np.random.uniform()
            x_copy[:, :, 2] = random_bright * x_copy[:, :, 2]
            x_copy[:, :, 2][x_copy[:, :, 2]>255] = 255 # clip to 0-255

            x_copy = sess.run(tf.image.hsv_to_rgb(x_copy))

        self.x = np.vstack((self.x, x_copy))
        self.y = np.vstack((self.y, y_copy))

    def _randomize(self):
        '''
            Randomize the dataset.
        '''
        indices = np.arange(self.num_points)
        np.random.shuffle(indices)
        self.x = x[indices]
        self.y = y[indices]
