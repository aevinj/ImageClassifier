import tensorflow as tf
from matplotlib import pyplot as plt
import cv2, os, subprocess
import numpy as np

class ImageClassifier:
    def __init__(self) -> None:
        self.DATA_DIRECTORY = 'data'
        # Supported image formats: jpeg, png, bmp - for keras
        # png causing libpng errors so disregarded as well
        self.IMG_EXTENSIONS = ['jpeg', 'bmp']
        self.data = None                        # Defined in load dataset function
        self.training_size = None               # Defined in split dataset function
        self.validation_size = None             # Defined in split dataset function
        self.testing_size = None                # Defined in split dataset function
        self.train = None                       # Defined in partition data function
        self.validate = None                  # Defined in partition data function
        self.test = None                        # Defined in partition data function

    def get_DATA_DIRECTORY(self):
        return self.DATA_DIRECTORY
    
    def get_IMG_EXTENSIONS(self):
        return self.IMG_EXTENSIONS

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
        data = tf.keras.utils.image_dataset_from_directory(self.get_DATA_DIRECTORY())
        data = data.map(lambda x, y: (x / 255, y))
        self.data = data
        # data_iterator = data.as_numpy_iterator()
        # batch = data_iterator.next()
        
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
        
        
        

        
                    
if __name__ == '__main__':
    ic = ImageClassifier()
    #ic.cleanDataset()
    ic.loadDataset()
    ic.split_and_partition_dataset()