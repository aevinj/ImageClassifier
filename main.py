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
        data_iterator = data.as_numpy_iterator()
        batch = data_iterator.next()
        print(batch[1])
        
                    
if __name__ == '__main__':
    ic = ImageClassifier()
    ic.cleanDataset()
    ic.loadDataset()