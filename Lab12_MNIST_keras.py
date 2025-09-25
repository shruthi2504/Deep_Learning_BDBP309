import tensorflow as tf
import keras
from keras import datasets, layers, models
from skimage.color import xyz_tristimulus_values
from sympy.physics.units import momentum
import os

os.environ['HTTP_PROXY']  = "http://245hsbd014%40ibab.ac.in:ibabstudent@proxy.ibab.ac.in:3128/"
os.environ['HTTPS_PROXY'] = "http://245hsbd014%40ibab.ac.in:ibabstudent@proxy.ibab.ac.in:3128/"

#load MNIST dataset
(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()


#add channel dimension(MNIST dataset has grayscale images and CNN expects 3D images)
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

#Normalize pixel values to [-1,1] like pytorch normalize(0.5,)
x_train = (x_train / 255.0 - 0.5) / 0.5
x_test = (x_test / 255.0 - 0.5) / 0.5

#class labels
classes = tuple(str(i) for i in range(10))
#spltting train into train and val set
x_train_partial = x_train[:50000]
y_train_partial = y_train[:50000]

x_val = x_train[50000:]
y_val = y_train[50000:]

#Define CNN
model = models.Sequential([
    layers.Conv2D(6,(5,5), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(16,(5,5), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10)
])

#compute model
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001,momentum=0.9),
                loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics = ['accuracy'])

#train model
history = model.fit(x_train_partial, y_train_partial, epochs=10, validation_data=(x_val, y_val),verbose=2)

#evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)
