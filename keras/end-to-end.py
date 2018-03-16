from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Flatten, Dense

class End2End:
	def __init__(self, img_size):
		self._build_model(img_size)

	def _build_model(self, img_size):
		'''
			Builds the end to end deep learning model for auto-steering.
			Args:
				img_size : tuple/list : image dimensions (H, W, C)
		'''
		
		self.model = Sequential()

		# Input layer
		self.model.add(BatchNormalization(), input_shape=img_size)
		
		# Strided 5x5 convolutions
		self.model.add(Conv2D(3, (5, 5), activation='relu', strides=(2, 2), padding='valid'))
		self.model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2), padding='valid'))
		self.model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2), padding='valid'))

		# Non-strided 3x3 convolutions
		self.model.add(Conv2D(48, (3, 3), activation='relu'))
		self.model.add(Conv2D(64, (3, 3), activation='relu'))

		# Flatten
		self.model.add(Flatten())

		# Dense layers
		self.model.add(Dense(1024, activation='elu'))
		self.model.add(Dense(256, activation='elu'))
		self.model.add(Dense(64, activation='elu'))
		self.model.add(Dense(8, activation='elu'))
		self.model.add(Dense(1, activation='tanh'))

	def _compile(self, optimizer='adam', loss='mean_squared_error'):
		self.model.compile(optimizer=optimizer, loss=loss)

	def train(self, path):
		pass

