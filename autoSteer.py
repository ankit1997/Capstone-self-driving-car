'''

Implementation of End to End Learning for Self-Driving Cars by Mariusz Bojarski et al.
@author: Ankit Bindal

'''

import os
import tensorflow as tf
from helper import CnnParams, DenseParams, Data

class AutoSteer:
    def __init__(self, image_shape):

        # End-to-End self driving pipeline

        # 1. Build the model
        self.build_model(image_shape)

        # 2. Define loss
        self.define_loss()

        # 3. Define optimizer
        self.define_optimizer()
        
        # Model checkpoint file
        model_dir="saved_model"
        os.makedirs(model_dir, exist_ok=True)
        self.model_file = os.path.join(model_dir, "model.ckpt")

        self.saver = tf.train.Saver()

    def build_model(self, image_shape):
        '''
        Builds the end to end self driving car model.
        The primary input is the image of the front view of road and the output is steering value.
        '''
        
        # Reset the tensorflow graph before building the model.
        tf.reset_default_graph()

        # Define input/output placeholders
        self.images = tf.placeholder(tf.float32, shape=(None, *image_shape), name='camera-view')
        self.steering = tf.placeholder(tf.float32, shape=(None, 1), name='correct-steering')
        self.keep_prob = tf.placeholder(tf.float32, name='keep-prob')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # Define convolution architecture parameters.
        # Parameters : (kernel size, number of out channels, stride, padding)
        cnn_parameters = [
            CnnParams(3, 16, 1, "SAME", tf.nn.elu),
            CnnParams(3, 32, 1, "VALID", tf.nn.elu),
            CnnParams(3, 64, 1, "SAME", tf.nn.elu),
            CnnParams(3, 128, 1, "SAME", tf.nn.elu),
        ]

        # Define dense layers parameters.
        # Parameters : (num_units, activation, *dropout(keep prob))
        dense_parameters = [
            DenseParams(256, tf.nn.elu, None),
            DenseParams(128, tf.nn.elu, self.keep_prob),
            DenseParams(64, tf.nn.elu, None),
            DenseParams(32, tf.nn.elu, self.keep_prob),
            DenseParams(8, tf.nn.elu, None),
            DenseParams(1, tf.nn.tanh, None)
        ]

        # Batch normalization layer
        layer = self.images
        layer = tf.layers.batch_normalization(layer, training=self.is_training)

        # Build CNN layers
        for i, param in enumerate(cnn_parameters):
            layer = self.cnn_layer(layer, param, i)

        # Flat the volume for dense layers
        layer = self.flatten_layer(layer)

        # Build Dense layers
        for i, param in enumerate(dense_parameters):
            layer = self.dense_layer(layer, param, i)

        self.prediction = layer

    def define_loss(self):
        self.loss = tf.losses.mean_squared_error(labels=self.steering, predictions=self.prediction)

    def define_optimizer(self, learning_rate=0.001):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def cnn_layer(self, inp, cnn_param, layer_num):
        '''
            Applies a convolution layer to input volume.
            Args:
                inp : Input volume Tensor
                cnn_param : a CnnParams object
                layer_num : Layer number (for scope and var name)
            Returns:
                Output volume
        '''

        inp_channels = inp.shape[-1].value

        with tf.variable_scope("Convolution-layer-{}".format(layer_num)):
            w = self.weight("cnn-W{}".format(layer_num), 
                            (cnn_param.kernel_size, cnn_param.kernel_size, inp_channels, cnn_param.filters))
            b = self.weight("cnn-B{}".format(layer_num), 
                            (cnn_param.filters))

            conv = tf.nn.conv2d(inp, w, [1, cnn_param.stride, cnn_param.stride, 1], cnn_param.padding) + b
            if cnn_param.activation:
                conv = cnn_param.activation(conv)

        return conv

    def dense_layer(self, inp, dense_param, layer_num):
        '''
            Applies a dense layer to input tensor.
            Args:
                inp : Input Tensor
                dense_param : a DenseParams object
                layer_num : Layer number (for scope and var name)
            Returns:
                Output tensor
        '''

        inp_units = inp.shape[-1].value

        with tf.variable_scope("Dense-layer-{}".format(layer_num)):
            w = self.weight("dense-W{}".format(layer_num), 
                            (inp_units, dense_param.units))
            b = self.bias("dense-B{}".format(layer_num), 
                        (dense_param.units))

            out = tf.matmul(inp, w) + b

            if dense_param.activation:
                out = dense_param.activation(out)

            if dense_param.dropout is not None:
                out = tf.nn.dropout(out, keep_prob=dense_param.dropout)

        return out

    def flatten_layer(self, inp):
        '''
            Flattens an input volume.
            Args:
                inp : Input volume tensor of shape (B, H, W, C)
            Returns:
                Reshaped `inp` : (B, H*W*C)
        '''
        num_units = (inp.shape[1] * inp.shape[2] * inp.shape[3]).value
        return tf.reshape(inp, [-1, num_units], name='flatten-layer')

    def weight(self, name, shape):
        return tf.get_variable(name, shape)

    def bias(self, name, shape):
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))

    def train(self, x, y, epochs=1000, log_step=10, resume=False):
        '''
            Trains the end-to-end model on provided data.
            Args:
                x : Numpy array of shape (Batch, Height, Width, Channels)
                y : Numpy array of shape (Batch, 1)
        '''

        data = Data(x, y)

        with tf.Session() as session:
            if resume:
                if os.path.isfile(self.model_file):
                    self.saver.restore(session, self.model_file)
                else:
                    print("No previous checkpoint file found, restarting training...")

            session.run(tf.global_variables_initializer())

            for e in range(epochs):
                avg_loss = 0.0
                for batch_x, batch_y in data.next_batch():
                    batch_loss, _ = session.run([self.loss, self.train_step], 
                                    feed_dict={self.images: x, self.steering: y, self.is_training: True, self.keep_prob: 0.8})
                    avg_loss += batch_loss

                if e%log_step == 0:
                    print("Average Mean squared loss for epoch {} = {}.".format(e, avg_loss))
                    
                    # Save the model state
                    fname = self.saver.save(session, self.model_file)
                    print("Session saved in {}".format(fname))

        print("Training complete!")

    def predict(self, x):
        '''
            Perform inference on given batch of images.
            Args:
                x : Batch of images of shape (B, H, W, C)
            Returns:
                Steering predictions (B, 1)
        '''

        with tf.Session() as session:
            self.saver.restore(session, self.model_file)
            prediction = session.run(self.prediction, feed_dict={self.images: x})

        return prediction

    def __str__(self):
        s = ""
        variables = vars(self)
        for var in variables:
            s = s + "{} : {}\n".format(var, self.__getattribute__(var))
        return s.strip()

# Testing
if __name__ == '__main__':
    autoSteer = AutoSteer((256, 256, 4))
    print(autoSteer)