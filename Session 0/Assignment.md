## What are Channels and Kernels (according to EVA)?
https://ai.stackexchange.com/questions/9751/what-is-the-concept-of-channels-in-cnns
https://www.quora.com/What-do-channels-refer-to-in-a-convolutional-neural-network

Components with which an image is formed are called as channels. Every kernel in CNN will move in XY direction only. Every pixel has three(in case of RGB) values for an image. Every value will have different kind of information stored about the same object in the images which can help us differentiating or extracting textures gradients from the images. In short by sliding across different pixel values across three channels CNN is trying to learn features in XY direction from all the channels together.

Kernels are feature extractors. They do this by sliding across the image in the specified matrix form (3x3|5x5....) to extract convolved feature. kernel is nothing but a filter that is used to extract the features from the images. It moves over the input data, performs the dot product with the sub-region of input data, and gets the output as the matrix of dot products.  In short, the kernel is used to extract high-level features like edges from the image.


## Why should we (nearly) always use 3x3 kernels?
## How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)
## How are kernels initialized? 
https://stats.stackexchange.com/questions/200513/how-to-initialize-the-elements-of-the-filter-matrix
https://stats.stackexchange.com/questions/267807/cnn-kernels-updates-initialization
https://www.quora.com/How-are-convolutional-filters-kernels-initialized-and-learned-in-a-convolutional-neural-network-CNN
https://ai.stackexchange.com/questions/5092/how-are-kernels-input-values-initialized-in-a-cnn-network#:~:text=The%20kernels%20are%20usually%20initialized,are%20many%20different%20initialization%20strategies.&text=For%20specific%20types%20of%20kernels,that%20seem%20to%20perform%20well.

## What happens during the training of a DNN?

Mainly Two things happen in CNN
###### 1)Feature Learning: 
	Series of convolutional layers that convolve with a multiplication or other dot product
###### 2)Classification: 
Activation function is commonly a RELU layer, and is subsequently followed by additional convolutions such as pooling layers, fully connected layers and normalization layers, referred to as hidden layers because their inputs and outputs are masked by the activation function and final convolution.
The final convolution, in turn, often involves backpropagation in order to more accurately weight the end product.




