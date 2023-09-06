# Multiclass Image Classification

Made solely by Aevin Jais

## Project Description

This project utilises the Keras sequential model to create a custom deep convoluted neural network in order to classify images into multiple classes. In this specific case, there are three classes: Cats, Dogs and Fish.

I utilised an initial model composed of 3 convolutional layers in order to create the model. Upon training for 10 epochs, ```the val_accuracy converged to 73%.``` This was a solid start.

From here I increased the complexity of the model and added more 2D convolutional layers. Initially, I trained for 10 epochs but noticed the val_accuracy was dropping around 8, indicating overfitting was occurring. So I reran on 8 epochs and got the following:

![Screenshot 2023-09-06 213143](https://github.com/aevinj/ImageClassifier/assets/64698098/973e2b79-16bf-4490-8d36-54218f4b1eb9)
X-Axis: epochs | Y-Axis: accuracy (max is 1)

```This change resulted in an improved accuracy of 76%.```

After, this I altered the learning rate via my optimizer (Adam) from the default of 0.001 to 0.0001. Here are the results:

![Screenshot 2023-09-06 215634](https://github.com/aevinj/ImageClassifier/assets/64698098/b6cd9e03-0df0-4a58-a99e-0603dda8e420)
X-Axis: epochs | Y-Axis: accuracy (max is 1)

```This change resulted in an improved accuracy of 81%.```

However, I believe there to be some degree of overfitting being introduced even in this model given that the gradient of val_accuracy plateaus around epoch 6. Nevertheless, the val_accuracy of this model was 81% and my highest-achieving model.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#Usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Features

main.py offers the ability to:
 - Create the model (given that you have the necessary libraries installed - and preferably have your GPU enabled for TensorFlow) (using the model template in the code of course)
 - Build the model
 - Train the model
 - Evaluate the model
 - Test on data that is unseen to the model
 - Load an existing model (avoids the need to create a model)

## Installation

***Python 3.9.4 was used to run this code. I suggest you use Python 3.9.* as well****

There is a requirements.txt file in the root directory. Use this along with pip in order to install the necessary libraries:

```pip install -r requirements.txt```

***NOTE: requirements.txt will not enable GPU usage. You have to do that yourself. Read below:***

This project was made on my laptop. I have a mobile RTX 3060, therefore I was able to train my models in minutes as opposed to hours. If you do not configure GPU usage or have a GPU usage the code will still work but just slowly. 

To enable GPU usage, follow this guide on installing WSL on Windows machines: https://www.tensorflow.org/install/pip#step-by-step_instructions

Once installing WSL, you will need to run the following lines:

```
conda activate tf
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
```
## Usage

***To be implemented***

## Technologies Used

Language:

Python

Libraries:

 - TensorFlow
 - matplotlib
 - cv2
 - numpy

## Contributing

If you're open to contributing to this project please contact me via email: ajjaevinjais@gmail.com.

## License

None

## Contact Information

Email: ajjaevinjais@gmail.com
IG: aevin.j

## Acknowledgments
@NicholasRenotte
@KGPTalkie
@SimplilearnOfficial
