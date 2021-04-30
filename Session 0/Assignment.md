## What are Channels and Kernels (according to EVA)?

Components with which an image is formed are called as channels. Every kernel in CNN will move in XY direction only. Every pixel has three(in case of RGB) values for an image. Every value will have different kind of information stored about the same object in the images which can help us differentiating or extracting textures gradients from the images. In short by sliding across different pixel values across three channels CNN is trying to learn features in XY direction from all the channels together.

Kernels are feature extractors. They do this by sliding across the image in the specified matrix form (3x3|5x5....) to extract convolved feature. kernel is nothing but a filter that is used to extract the features from the images. It moves over the input data, performs the dot product with the sub-region of input data, and gets the output as the matrix of dot products.  In short, the kernel is used to extract high-level features like edges from the image.

https://ai.stackexchange.com/questions/9751/what-is-the-concept-of-channels-in-cnns            
https://www.quora.com/What-do-channels-refer-to-in-a-convolutional-neural-network             

## Why should we (nearly) always use 3x3 kernels?
5x5 creates 25 parameters whereas the replacement 3x3 does the same in two layers with 9 features in each layer. However, this is not always true that 3x3 is effective there are cases when 5x5 needed 3 layers of 3x3 kernels to match the performance of the former.
2x2 doesnt have axis of symmetry to perform effectively.
Above 5x5 also requires more computations than a simple multiple 3x3 layers

## How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)

199x199 | 3x3 > 197x197         
197x197 | 3x3 > 195x195         
195x195 | 3x3 > 193x193         
193x193 | 3x3 > 191x191         
191x191 | 3x3 > 189x189         
189x189 | 3x3 > 187x187         
187x187 | 3x3 > 185x185         
185x185 | 3x3 > 183x183         
183x183 | 3x3 > 181x181         
181x181 | 3x3 > 179x179         
179x179 | 3x3 > 177x177         
177x177 | 3x3 > 175x175         
175x175 | 3x3 > 173x173         
173x173 | 3x3 > 171x171         
171x171 | 3x3 > 169x169         
169x169 | 3x3 > 167x167         
167x167 | 3x3 > 165x165         
165x165 | 3x3 > 163x163         
163x163 | 3x3 > 161x161         
161x161 | 3x3 > 159x159         
159x159 | 3x3 > 157x157         
157x157 | 3x3 > 155x155         
155x155 | 3x3 > 153x153         
153x153 | 3x3 > 151x151         
151x151 | 3x3 > 149x149         
149x149 | 3x3 > 147x147         
147x147 | 3x3 > 145x145         
145x145 | 3x3 > 143x143         
143x143 | 3x3 > 141x141         
141x141 | 3x3 > 139x139         
139x139 | 3x3 > 137x137         
137x137 | 3x3 > 135x135         
135x135 | 3x3 > 133x133         
133x133 | 3x3 > 131x131         
131x131 | 3x3 > 129x129         
129x129 | 3x3 > 127x127         
127x127 | 3x3 > 125x125         
125x125 | 3x3 > 123x123         
123x123 | 3x3 > 121x121         
121x121 | 3x3 > 119x119         
119x119 | 3x3 > 117x117         
117x117 | 3x3 > 115x115         
115x115 | 3x3 > 113x113         
113x113 | 3x3 > 111x111         
111x111 | 3x3 > 109x109         
109x109 | 3x3 > 107x107         
107x107 | 3x3 > 105x105         
105x105 | 3x3 > 103x103         
103x103 | 3x3 > 101x101         
101x101 | 3x3 > 99x99         
99x99   | 3x3 > 97x97         
97x97   | 3x3 > 95x95         
95x95   | 3x3 > 93x93         
93x93   | 3x3 > 91x91         
91x91   | 3x3 > 89x89         
89x89   | 3x3 > 87x87                
87x87   | 3x3 > 85x85         
85x85   | 3x3 > 83x83         
83x83   | 3x3 > 81x81         
81x81   | 3x3 > 79x79         
79x79   | 3x3 > 77x77         
77x77   | 3x3 > 75x75         
75x75   | 3x3 > 73x73         
73x73   | 3x3 > 71x71         
71x71   | 3x3 > 69x69         
69x69   | 3x3 > 67x67         
67x67   | 3x3 > 65x65         
65x65   | 3x3 > 63x63         
63x63   | 3x3 > 61x61         
61x61   | 3x3 > 59x59         
59x59   | 3x3 > 57x57         
57x57   | 3x3 > 55x55         
55x55   | 3x3 > 53x53         
53x53   | 3x3 > 51x51         
51x51   | 3x3 > 49x49         
49x49   | 3x3 > 47x47                  
47x47   | 3x3 > 45x45         
45x45   | 3x3 > 43x43         
43x43   | 3x3 > 41x41         
41x41   | 3x3 > 39x39         
39x39   | 3x3 > 37x37         
37x37   | 3x3 > 35x35         
35x35   | 3x3 > 33x33         
33x33   | 3x3 > 31x31                  
31x31   | 3x3 > 29x29         
29x29   | 3x3 > 27x27         
27x27   | 3x3 > 25x25         
25x25   | 3x3 > 23x23         
23x23   | 3x3 > 21x21         
21x21   | 3x3 > 19x19         
19x19   | 3x3 > 17x17         
17x17   | 3x3 > 15x15         
15x15   | 3x3 > 13x13         
13x13   | 3x3 > 11x11         
11x11   | 3x3 > 9x9         
9x9     | 3x3 > 7x7         
7x7     | 3x3 > 5x5         
5x5     | 3x3 > 3x3         
3x3     | 3x3 > 1x1         

## How are kernels initialized? 
Before the advances in optimization techniques and activation of non linearities, kernel is initializes from a random distribution. With advances, all the advances we are able to train convolutional neural networks from a randomized initialization.

Weights of the backpropagation will be randomly initialized such that the mean of the distribution weights is 0 and standard deviation of 1.

###### Potential Problems with Weight Initializations
1) If the weights are too small, then the signal shrinks as it passes through each layer. Ends up with very small value to use.      
2) If the weights are too large, then the signal grows too large as it passes through each layer, which will bevery large by the time it ends    

Xavier Initialization takes care of this problem. For more understanding on Xavier initialization below blog and vlog can be used.      
https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization     
https://www.youtube.com/watch?v=8krd5qKVw-Q    
Paper on Xavier Initialization : http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

https://stats.stackexchange.com/questions/200513/how-to-initialize-the-elements-of-the-filter-matrix
https://stats.stackexchange.com/questions/267807/cnn-kernels-updates-initialization
https://www.quora.com/How-are-convolutional-filters-kernels-initialized-and-learned-in-a-convolutional-neural-network-CNN
https://ai.stackexchange.com/questions/5092/how-are-kernels-input-values-initialized-in-a-cnn-network#:~:text=The%20kernels%20are%20usually%20initialized,are%20many%20different%20initialization%20strategies.&text=For%20specific%20types%20of%20kernels,that%20seem%20to%20perform%20well.

## What happens during the training of a DNN?

<img src="images/1_vkQ0hXDaQv57sALXAJquxA.jpeg"  />                            
Mainly Two things happen in CNN      

###### 1) Feature Learning:                                                                 
Series of convolutional layers that convolve with a multiplication or other dot product             
           
###### 2) Classification:                                                         
Activation function is commonly a RELU layer, and is subsequently followed by additional convolutions such as pooling layers, fully connected layers and normalization layers, referred to as hidden layers because their inputs and outputs are masked by the activation function and final convolution.                                                                       
The final convolution, in turn, often involves backpropagation in order to more accurately weight the end product.                                                    




