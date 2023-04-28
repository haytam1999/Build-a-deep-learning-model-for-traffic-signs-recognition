# Build-a-machine-learning-model-for-traffic-signs-recognition
We are going to build a machine learning model with which we can do traffic sign detection and recognition, from a video stream captured by the car camera.

The training of this model is based on the convolutional neural networks (CNN) which allows to extract the features and the characteristics of an image on which it is based to make the prediction during the part of the real recognition.
To implement the CNN, we will use the TensorFlow Keras API, which facilitates the construction of the different layers of the neural network.

For operations on images, there is no better than the OpenCV library which offers numerous APIs for the manipulation of video and image streams in real time, and thus for the various transformations that will be applied to them.

Model training will be done with the 'train, test, validate split' method found in the Scikit Learn library. With it, the dataset is divided into 3 parts:

The large 'train' part: it contains the images that the neural network will use to extract the features and adjust the weights and biases of its links.
The 'validation' part: it contains the images with which, and during training, the model will test its learning and will adjust its parameters to have better accuracy after each epoch.
The 'test' part: this part contains the images that the model will test, after the end of its training, its accuracy and its ability to predict new images never seen by it.

Finally, we use the Numpy and Pandas libraries for manipulating arrays and dataframes, and MatPlotLib and Seaborn for displaying graphs and curves.



For more details, check the code provided here...
##########################################
Test Accuracy obtained: 0.9719298481941223
##########################################
