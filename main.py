import tensorflow as tf
from tensorflow import keras
from keras import models as tfModels
from keras import layers as tfLayers
from keras import utils as tfUtils
from matplotlib import pyplot as plt
import cv2, os, subprocess
import numpy as np

class ImageClassifier:
    def __init__(self) -> None:
        self.DATA_DIRECTORY = 'data'
        self.LOGS_DIRECTORY = 'logs'
        # Supported image formats: jpeg, png, bmp - for keras
        # png causing libpng errors so disregarded as well
        self.IMG_EXTENSIONS = ['jpeg', 'bmp']
        self.data = None                        # Defined in load dataset function
        self.training_size = None               # Defined in split dataset function
        self.validation_size = None             # Defined in split dataset function
        self.testing_size = None                # Defined in split dataset function
        self.train = None                       # Defined in partition data function
        self.validate = None                    # Defined in partition data function
        self.test = None                        # Defined in partition data function
        self.model = None                       # Defined in build model function

    def get_DATA_DIRECTORY(self):
        return self.DATA_DIRECTORY
    
    def get_IMG_EXTENSIONS(self):
        return self.IMG_EXTENSIONS
    
    def get_LOGS_DIRECTORY(self):
        return self.LOGS_DIRECTORY
    
    def set_model(self, model):
        self.model = model

    def get_file_extension(self, file_path):
        # Split the file path into its base name and extension
        base_name, extension = os.path.splitext(file_path)
        
        # Remove the leading dot (.) from the extension
        extension = extension.lstrip('.')
        return extension

    def cleanDataset(self):
        for image_class in os.listdir(self.get_DATA_DIRECTORY()): 
            for image in os.listdir(os.path.join(self.get_DATA_DIRECTORY(), image_class)):
                image_path = os.path.join(self.get_DATA_DIRECTORY(), image_class, image)
                try:
                    # To check if the file can be opened and has the right extension :)                    
                    if image.endswith('.jpg'):
                        img = cv2.imread(image_path)
                        cv2.imwrite(os.path.join(self.get_DATA_DIRECTORY(), image_class, image[:-3] + ".jpeg"), img)
                        
                    ext = self.get_file_extension(image_path)
                    if ext not in self.get_IMG_EXTENSIONS():
                        os.remove(image_path)
                except Exception as e:
                    os.remove(image_path)
                    
    def loadDataset(self):
        data = tfUtils.image_dataset_from_directory(self.get_DATA_DIRECTORY(), label_mode='categorical')
        data = data.map(lambda x, y: (x / 255, y))
        self.data = data
        data_iterator = data.as_numpy_iterator()
        batch = data_iterator.next()
        
    def split_and_partition_dataset(self):
        self.splitDataset(len(self.data))
        self.partitionDataset()
        
    def splitDataset(self, total_batches):
        training_split = 0.7
        validation_split = 0.2
        testing_split = 0.1

        # Calculate the number of batches for each split
        training_batches = int(total_batches * training_split)
        validation_batches = int(total_batches * validation_split)
        testing_batches = int(total_batches * testing_split)

        # Ensure that the splits add up to total_batches
        while training_batches + validation_batches + testing_batches < total_batches:
            training_batches += 1

        # Adjust the splits if they exceed total_batches
        while training_batches + validation_batches + testing_batches > total_batches:
            if training_batches > 0:
                training_batches -= 1
            elif validation_batches > 0:
                validation_batches -= 1
            else:
                testing_batches -= 1

        # Assign the calculated sizes to instance variables
        self.training_size = training_batches
        self.validation_size = validation_batches
        self.testing_size = testing_batches
        
    def partitionDataset(self):
        self.train = self.data.take(self.training_size)
        self.validate = self.data.skip(self.training_size).take(self.validation_size)
        self.test = self.data.skip(self.training_size + self.validation_size).take(self.testing_size)
        
    def buildModel(self):
        model = tfModels.Sequential()
        activation = 'sigmoid'
        model.add(tfLayers.Conv2D(32, 3, activation = activation, padding = 'same', input_shape = (256, 256, 3)))
        model.add(tfLayers.BatchNormalization())

        model.add(tfLayers.Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
        model.add(tfLayers.BatchNormalization())
        model.add(tfLayers.MaxPooling2D())

        model.add(tfLayers.Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
        model.add(tfLayers.BatchNormalization())

        model.add(tfLayers.Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
        model.add(tfLayers.BatchNormalization()) 
        model.add(tfLayers.MaxPooling2D())

        model.add(tfLayers.Flatten())
        model.add(tfLayers.Dense(128, activation = activation, kernel_initializer = 'he_uniform'))
        model.add(tfLayers.Dense(3, activation = 'softmax'))
        model.compile('rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
    
    def trainModel(self):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.get_LOGS_DIRECTORY())
        training_history = self.model.fit(self.train, epochs=10, validation_data=self.validate, callbacks=[tensorboard_callback])
        self.saveModel()
        
        # fig = plt.figure()
        # plt.plot(training_history.history['loss'], color='teal', label='loss')
        # plt.plot(training_history.history['val_loss'], color='orange', label='val_loss')
        # fig.suptitle('Loss', fontsize=20)
        # plt.legend(loc="upper left")
        # plt.show()
        
    def testModel(self, img):
        resize = tf.image.resize(img, (256,256))
        yhat = self.model.predict(np.expand_dims(resize/255, 0))
        #print("yhat is ", yhat)
        class_index = np.argmax(yhat)
        #print("class index is ", class_index)
        
        if class_index == 0:
            print('Predicted class is cat\n')
        elif class_index == 1:
            print('Predicted class is dog\n')
        elif class_index == 2:
            print('Predicted class is fish\n')
            
    def saveModel(self):
        self.model.save('ImageClassifier.keras')
        
if __name__ == '__main__':
    ic = ImageClassifier()
    if os.path.exists('ImageClassifier.keras'):
        ic.set_model(tfModels.load_model('ImageClassifier.keras'))
        # print("fish test")
        # ic.testModel(img = cv2.imread('fish.jpeg'))
        # print("dog test")
        # ic.testModel(img = cv2.imread('dog.jpeg'))
        # print("cat test")
        # ic.testModel(img = cv2.imread('cat.jpeg'))
        # print("cat test")
        # ic.testModel(img = cv2.imread('cat2.jpeg'))
        # print("cat test")
        # ic.testModel(img = cv2.imread('cat3.jpeg'))
        print("fish test")
        ic.testModel(img = cv2.imread('fish2.jpeg'))
    else:
        ic.cleanDataset()
        ic.loadDataset()
        ic.split_and_partition_dataset()
        ic.buildModel()
        ic.trainModel()
        print("fish test")
        ic.testModel(img = cv2.imread('fish.jpeg'))
        print("dog test")
        ic.testModel(img = cv2.imread('dog.jpeg'))
        print("cat test")
        ic.testModel(img = cv2.imread('cat.jpeg'))
        print("cat test")
        ic.testModel(img = cv2.imread('cat2.jpeg'))
        print("cat test")
        ic.testModel(img = cv2.imread('cat3.jpeg'))
        print("fish test")
        ic.testModel(img = cv2.imread('fish2.jpeg'))