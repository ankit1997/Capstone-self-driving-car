'''

Implementation of End to End Learning for Self-Driving Cars by Mariusz Bojarski et al.
@author: Ankit Bindal

'''

import os
import sys
from tqdm import tqdm
import tensorflow as tf
from helper import CnnParams, DenseParams, Data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

        self.tensorboard_op = tf.summary.merge_all()

    def build_model(self, image_shape):
        '''
        Builds the end to end self driving car model.
        The primary input is the image of the front view of road and the output is steering command - left/right/straight.
        '''
        
        # Reset the tensorflow graph before building the model.
        tf.reset_default_graph()

        # Define input/output placeholders
        self.images = tf.placeholder(tf.float32, shape=(None, *image_shape), name='camera-view')
        self.steering = tf.placeholder(tf.int32, shape=(None, 1), name='correct-steering')
        self.keep_prob = tf.placeholder(tf.float32, name='keep-prob')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # Define convolution architecture parameters.
        # Parameters : (kernel size, number of out channels, stride, padding)
        self.cnn_parameters = [
            CnnParams(5, 16, 2, "VALID", tf.nn.elu),
            CnnParams(5, 32, 2, "VALID", tf.nn.elu),
            CnnParams(5, 64, 2, "VALID", tf.nn.elu),
            CnnParams(3, 128, 1, "VALID", tf.nn.elu),
        ]

        # Define dense layers parameters.
        # Parameters : (num_units, activation, *dropout(keep prob))
        self.dense_parameters = [
            DenseParams(1024, tf.nn.elu, None),
            DenseParams(256, tf.nn.elu, self.keep_prob),
            DenseParams(64, tf.nn.elu, None),
            DenseParams(8, tf.nn.elu, self.keep_prob),
            DenseParams(1, tf.nn.tanh, None)
        ]

        # Batch normalization layer
        layer = self.images
        layer = tf.layers.batch_normalization(layer, training=self.is_training)

        # Build CNN layers
        for i in range(len(self.cnn_parameters)):
            layer = self.cnn_layer(layer, i)

        # Flat the volume for dense layers
        layer = self.flatten_layer(layer)

        # Build Dense layers
        for i in range(len(self.dense_parameters)):
            layer = self.dense_layer(layer, i)

        self.prediction = layer * 90.0

        # Get image showing features extracted by CNN
        self.visualize = self.visualBackProp()

    def define_loss(self):
        # self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.steering, logits=self.prediction)
        self.loss = tf.losses.mean_squared_error(labels=self.steering, predictions=self.prediction)
        tf.summary.scalar("Mean-squared-loss", self.loss) # for tensorboard

    def define_optimizer(self, learning_rate=0.00001):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def cnn_layer(self, inp, layer_num):
        '''
            Applies a convolution layer to input volume.
            Args:
                inp : Input volume Tensor
                layer_num : Layer number (for scope and var name)
            Returns:
                Output volume
        '''
        cnn_param = self.cnn_parameters[layer_num]
        inp_channels = inp.shape[-1].value

        with tf.variable_scope("Convolution-layer-{}".format(layer_num)):
            w = self.weight("cnn-W{}".format(layer_num), 
                            (cnn_param.kernel_size, cnn_param.kernel_size, inp_channels, cnn_param.filters))
            b = self.weight("cnn-B{}".format(layer_num), 
                            (cnn_param.filters))

            conv = tf.nn.conv2d(inp, w, [1, cnn_param.stride, cnn_param.stride, 1], cnn_param.padding) + b
            if cnn_param.activation:
                conv = cnn_param.activation(conv)

            # name the tensor
            conv = tf.identity(conv, name='layer-output')

        return conv

    def dense_layer(self, inp, layer_num):
        '''
            Applies a dense layer to input tensor.
            Args:
                inp : Input Tensor
                layer_num : Layer number (for scope and var name)
            Returns:
                Output tensor
        '''
        dense_param = self.dense_parameters[layer_num]
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

    def visualBackProp(self):
        '''
            Implementation of VisualBackProp: efficient visualization of CNNs.
        '''

        # Get names of output op of all convolution layers
        operations = tf.get_default_graph().get_operations()
        op_names = [op.name for op in operations]
        cnn_op_names = list(filter(lambda name: name.startswith("Convolution-layer") and name.endswith("layer-output"), op_names))
        num_cnn_layers = len(cnn_op_names)

        # Just verifying that all layers were built successfully.
        assert num_cnn_layers == len(self.cnn_parameters)

        # Get tensor operations of all conv layers
        cnn_ops = [tf.get_default_graph().get_tensor_by_name(cnn_op_name+":0") for cnn_op_name in cnn_op_names]

        with tf.variable_scope("VisualBackProp"):
            # calculate average feature map
            average_fmaps = [tf.expand_dims(tf.reduce_mean(cnn_op, axis=-1), axis=-1) for cnn_op in cnn_ops]

            # up-scale last average feature map
            previous = self._visual_back_prop_step(average_fmaps[-1], average_fmaps[-2], len(average_fmaps)-1)
            
            for i in range(len(average_fmaps)-2, 0, -1):
                pointwise_mul = tf.multiply(previous, average_fmaps[i], name='pointwise-mul')
                previous = self._visual_back_prop_step(pointwise_mul, average_fmaps[i-1], i)

            # up-scale to original image
            mask = self._visual_back_prop_step(previous, self.images, 0)

            # normalize mask to range [0, 1]
            with tf.variable_scope("normalize-mask"):
                normalized_mask = tf.div(tf.subtract(mask, tf.reduce_min(mask)), tf.subtract(tf.reduce_max(mask), tf.reduce_min(mask)))

        return normalized_mask

    def _visual_back_prop_step(self, current_map, previous_map, layer_num):
        '''
            Helper function for deconvolution operation of visualbackprop model.
        '''

        cnn_param = self.cnn_parameters[layer_num]
        k = cnn_param.kernel_size
        s = cnn_param.stride
        
        return tf.nn.conv2d_transpose(current_map,
                filter=tf.constant(1.0, shape=[k, k, 1, 1], name="w-deconv-{}".format(layer_num), dtype=tf.float32),
                output_shape=[tf.shape(previous_map)[0], previous_map.shape[1], previous_map.shape[2], 1],
                strides=[1, s, s, 1],
                padding=cnn_param.padding)
        
    def train(self, path, epochs=100, log_step=10, resume=False):
        '''
            Trains the end-to-end model on provided data.
            Args:
                path : Path to dataset directory
        '''

        data = Data(path)

        with tf.Session() as session:
            writer = tf.summary.FileWriter("for_tensorboard", session.graph)

            resumed = False
            if resume:
                try:
                    self.saver.restore(session, self.model_file)
                    resumed = True
                    print("Resuming previous training...")
                except:
                    print("No previous checkpoint file found, restarting training...")
            
            if not resumed:
                session.run(tf.global_variables_initializer())

            try:
                for e in range(epochs):
                    loss_sum = 0.0
                    for batch_x, batch_y in data.next_batch():
                        batch_loss, _, tb_op = session.run([self.loss, self.train_step, self.tensorboard_op], 
                                        feed_dict={self.images: batch_x, self.steering: batch_y, self.is_training: True, self.keep_prob: 0.8})
                        loss_sum += batch_loss
                    
                    print("Sum of Mean squared losses for epoch {} = {}.".format(e, loss_sum))
                    
                    if e%10 == 0:
                        writer.add_summary(tb_op, e)

                        # Save the model state
                        fname = self.saver.save(session, self.model_file)
                        print("Session saved in {}".format(fname))

            except KeyboardInterrupt:
                pass

            writer.close()

        print("Training complete!")

        self.evaluate(data)

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
            prediction = session.run(self.prediction, 
                                    feed_dict={self.images: x, self.keep_prob: 1.0, self.is_training: False})

        return prediction

    def evaluate(self, data):
        avg_loss = 0.0
        c=0
        with tf.Session() as session:
            self.saver.restore(session, self.model_file)
            for batch_x, batch_y in data.next_batch(training=False):
                batch_loss = session.run(self.loss, 
                    feed_dict={self.images: batch_x, self.steering: batch_y, self.is_training: False, self.keep_prob: 1.0})
                avg_loss = avg_loss + batch_loss
                c+=1
            print("Average loss on given test data: {}".format(avg_loss/c))

    def get_visual_mask(self, x):
        '''
            Get visual mask according to VisualBackProp algorithm.
        '''
        with tf.Session() as session:
            self.saver.restore(session, self.model_file)
            prediction = session.run(self.visualize, 
                                    feed_dict={self.images: x, self.is_training: False})

        return prediction

    def weight(self, name, shape):
        return tf.get_variable(name, shape)

    def bias(self, name, shape):
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))

    def __str__(self):
        s = ""
        variables = vars(self)
        for var in variables:
            s = s + "{} : {}\n".format(var, self.__getattribute__(var))
        return s.strip()

# Testing
if __name__ == '__main__':
    autoSteer = AutoSteer((80, 320, 3))
    # autoSteer.train(sys.argv[1], epochs=1000, resume=True)

    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    fname = "/home/malcolm/Desktop/udacity-self-driving-car-data/IMG/center_2018_03_05_18_44_35_920_flipped.jpg"
    img = np.array(Image.open(fname))
    img = img[60: 140, :, :]
    img = np.expand_dims(img, axis=0)

    # with tf.Session() as session:
    #     img = session.run(tf.image.adjust_brightness(img, 0.3))

    mask = autoSteer.get_visual_mask(img)
    print(mask.shape)

    out = autoSteer.predict(img)
    print(out/90.0)
    img = np.squeeze(img)

    print(img.shape)
    
    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    plt.imshow(np.squeeze(mask))

    plt.show()