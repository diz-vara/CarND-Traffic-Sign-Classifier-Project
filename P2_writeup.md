**Traffic Sign Recognition** 


---



[//]: # (Image References)

[image1]: ./classes.png "Sign classes"
[image2]: ./motionBlur.png "Motion blur"
[image3]: ./rotations.png "Rotations"
[image4]: ./errors.png "errors"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/diz-vara/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration


The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples (39209 samples including validation)
* The size of test set is 12630 samples
* The shape of a traffic sign image is 32x32x3 (3 color planes of 32x32 pixels)
* The number of unique classes/labels in the data set is 43


The code for this step is contained in the X code cell of the IPython notebook.  

One example of each class shown, with number of samples in the header

![alt text][image1]

Without any additional analysis we can see, that the dataset is inbalanced, with number of samples of
some classes over 2000 (2010 for speed limit 50), and for other signs very low (180 for speed limit 20).

This distibution may be significant in a final product as a Bayesian prior of sign class, but for classification it is advisable to balance the dataset.

### Design and Test a Model Architecture

#### 1. Image pre-processing.

The code for this step is contained in the fourth code cell of the IPython notebook.

I decided not to convert images to the grayscale. Such a convertion seems good for model size 
and speed - but we certainly lose some valuable information.
More than that, 'simple' RGB2GRAY conversion can be represented as one additional convolution layer
with kernel size 1x1 and **fixed** (that means **not learnable**) coefficietns [0.299,0.587,0.114].
If we are sure we need, we can place such a 1x1 convolution layer at the begining of our
network - and **learn** coefficietns optimal for our task. 


Image normalization seems to be reasonable (but not necessary) step, because basically ANN performs linear 
operations that are sensitive to the mean and variation of the input variable.
Ususal practice is to normalize color channels separately (substracting channel mean and dividing by std or range).
But this approach may distort original color balance of the image. It may be good for removing color cast from red
sunset or blue night shots - but in case of this dataset it can affect recognition of red ('STOP') and blue road signs.
Therefore, I prefered 'global normalization' - removing mean value of the image and dividing by total range.

The main preprocessing step is the image set augmentation. I decided to add variants with distortions that we can
expect in real situaion (detecting sing from the moving vehicle) - motion blur and perspective transformation 
(corresponding to the object rotation in a 3D-space).

I decided to:
- balance dataset (equal number of images in each class);
- add some distortions even to the classes with large number of images.

I decided to extend dataset up to 4000 images per class (172000 training images). Only train set was augmened.
In case of cross-validation, augmentation was performed after splitting dataset into train and validation folds.

(*If you want to get a **very good** validation results, you may augment your dataset by a factor of 10 - and then
split it into train and validation set: most of the validation data will be just a variant of training images,
you'll get very good validation accuracy - but poor resuts for test and for real-world usage of your net.*)


For each new image were chosen randomly:
- motion blur: no motion (60%), moderate blur with 3x3 kernel (30%), strong blur with 5x5 kernel (10%)
- rotation around X (horizontal) axis: [0° 20°] (20° corresponds to the sign seen from below: it makes no sense to use negative angles,
	as normally we do not observe signs from above);
- rotation around Y (vertical) axis: [-20°  20°] (corrsponds to the visible rotation of the sign as we pass it);
- rotation around Z (coming 'out of the picture plane') axis: [-15°  15°] (corresponds to the errors in sign placement 
	and to the distortions introduced by wide-angle camera when we pass the sign).

The sixth code cell of the IPython notebook contains the code for augmenting the data set.


Here is an example of an original image and botion blur distortion applyed to that image:

![alt text][image2]

And here you can see rotations applyed to the same image

![alt text][image3]



#### 2. Dataset organization. 

First experiments were conducted using train, validation and test sets provided in the given file.
After several runs on different network arcitectures, I've noticed that didn't rise above 
the moderate value of 0.985 (it means 60-70 errors per original validation set with 4010 images).
I've displayed the images that were not classified correctly and noticed, that they contained
several sequences of the same sign in some 'extreme' conditions (e.g. over- or under-exposed).

Here is an example of sequence of valiation set errors:

![alt text][image4]

From that I've made a conclusion that trainig and validation set were not properly balanced (e.g., there were
no examples of over-exposured 30 sign in the training set). The solution was, according to wonderful
Andrew Ng's lecture ['Nuts and bolts of machine learning'](https://www.youtube.com/watch?v=F1ka6a13S9I), 
to:
- merge trainig and validation data into one 'Dev' set;
- reserve 10% of the set for pre-final test (DevTest set);
- use the rest of the data (DevValTrn set) for 5-fold cross-validation.

Dataset splitting was not completely random: I've preserved the same proportion (10% for DevTest and 20% for Validation)
for each class, merging and shuffling results afterwards. Data augmentation was performed on each fold of the
5-fold cross-validation.

 Original Test set was left intact and was not used untill the last step of the training.

DevTest set contained 3920 images. From the rest Dev set (35289 images), on each fold of the cross-validation 7039 images were
assigned to the Validation set, and 28520 - to the train set, which was autmented to 172000 images (4000 images per class).


The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

After model selection and parameter tuning, three best models were trained on the whole DevValTrn set and validated on DevTest set.
The best model was trained on united Dev set (once again - according to the advice from the ['Nuts and bolts'](https://www.youtube.com/watch?v=F1ka6a13S9I)) - 
and tested on the original test set.


#### 3. Model architecture.

First preliminary tests were conducted on 'lab' LeNet network.

Then I switched to the variant of the 'Multi-scale convolutional network' by Pierre Sermanet [1][2].
In '*real life* it was the first 'deep' network I used for sign recognition - first using  EBLearn framework by Sermanet [3],
then switching to more flexible (and supported) [Torch](http://torch.ch/) - Lua-based machine learning framework.

Starting from 'original' multi-scale network, some changes were introduced and tested, and
now my implementation has some differences from the original one:
- instead of 5x5 convolutions, I use sequence of two 3x3 covolutions (idea from [4]).
- I use **ReLU** activations between convolutional layers, and **tanh** after the last convolutional layer and between fully-connected layers;
- I use smaller layer sizes to reduce computational speed and to decrease the chance of overfitting.
- I use dropout [5] in fully-connected layer. 

(In 'production' torch code I use [PReLU](https://github.com/torch/nn/blob/master/doc/transfer.md#prelu) activations and [BatchNormalization] layer,
but I'm not ready to implement them in TF)

I've tested sevaral additional models with deeper layers, or additional fully-conneted layer. These models
 clearly overfitted (with zero train loss and validation loss a little bit larger then of chosen network).


The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Hyper Layer 1      	|					   							| 
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 30x30x4 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 28x28x8 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x8 					|
| RELU					|												|
| Hyper Layer 2      	|					   							| 
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 12x12x8 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| flatten to **Flat2**	    | 400x1    									|
| RELU					|												|
| Hyper Layer 3      	|					   							| 
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 3x3x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 1x1x32 	|
| RELU					|												|
| flatten to **Flat3**	    | 32x1    									|
| concatenate **Flat2**	and **Flat3**    | 432x1    					|
| Fully connected		| Input 432, output 120      					|
| dropout 0.5			|												|
| tanh					|												|
| Fully connected		| Input 120, output 43      					|
| Softmax				|	        									|
 


#### 4. Model training. include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.


Models were trained using ADAM optimizer with exponential learning rate decay (starting form 1e-1, it was reduced by a factor of 0.998 after 
70 iterations, so that on 50th epoch effective learnig rate was 8.5e-5);

The code for training the model is located in the eigth cell of the ipython notebook. 



#### 5. Discussion

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of .999
* validation set accuracy of .997
* test set accuracy of .961

I've planned to use iterative approach - but fished by using first architecture I've build for this project.
It is based on well-knows Pierre Sermaet multi-scale network [1][2], some changes (e.g. usage of ReLU and tanh
activations) I introduced and tested some years ago.

Desision to replace 5x5 convolution by a sequence of two 3x3 convolutions was influenced by [4]. It proved to be right desision: 
I've performed test with 'backward repacement'. and it showed the result that was 1% worse.



 

### Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### Conclusion

