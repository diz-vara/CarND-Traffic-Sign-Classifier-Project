**Traffic Sign Recognition** 


---



[//]: # (Image References)

[image1]: ./Pictures/classes.png "Sign classes"
[image2]: ./Pictures/motionBlur.png "Motion blur"
[image3]: ./Pictures/rotations.png "Rotations"
[image4]: ./Pictures/scale.png "scale"
[image5]: ./Pictures/light.png "light"

[image6]: ./newSigns/t/00000_[eu]_51-05-08.png "Traffic Sign 1"
[image7]: ./newSigns/t/00008_00019.png "Traffic Sign 2"
[image8]: ./newSigns/t/01450_[sl50]_00-35-73.png "Traffic Sign 3"
[image9]: ./newSigns/t/02712_(98)[IMG_2036]_03-21-40.png "Traffic Sign 4"
[image10]: ./newSigns/t/02842_[]_03-15-19.png "Traffic Sign 5"

[image11]: ./Pictures/newImages0.png "new0"
[image12]: ./Pictures/hard.png "new1"
[image13]: ./Pictures/HardTest.png "new1"

[image14]: ./Pictures/sl50.png "sl50"


Here is a link to my [project code](https://github.com/diz-vara/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration


The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples (39209 samples including validation)
* The size of test set is 12630 samples
* The shape of a traffic sign image is 32x32x3 (3 color planes of 32x32 pixels)
* The number of unique classes/labels in the data set is 43


The code for this step is contained in the code cells **2**, **3** and **4** of the IPython notebook.  

One example of each class shown, with number of samples in the header

![alt text][image1]

Without any additional analysis we can see, that the dataset is unbalanced, with number of samples of
some classes over 2000 (2010 for speed limit 50), and for other signs very low (180 for speed limit 20).

This distribution may be significant in a final product as a Bayesian prior of sign class, but for classification it is advisable to balance the dataset.

### Design and Test a Model Architecture

#### 1. Image pre-processing.

The code for this step is contained in the code cell **5** of the IPython notebook.

I decided not to convert images to the grayscale. Such a conversion seems good for model size 
and speed - but we certainly lose some valuable information.
More than that, 'simple' RGB2GRAY conversion can be represented as one additional convolution layer
with kernel size 1x1 and *fixed* (that means *not learnable*) coefficients [0.299,0.587,0.114].
If we are sure we need, we can place such a 1x1 3->1 convolution layer as the first layer of our
network - and **learn** coefficients optimal for our task. 


Image normalization seems to be reasonable (but not necessary) step, because basically ANN performs linear 
operations that are sensitive to the mean and variation of the input variable.
Usual practice is to normalize color channels separately (subtracting channel mean and dividing by variation  or range).
But this approach may distort original color balance of the image. It may be good for removing color cast from red
sunset or blue night shots - but in case of this dataset it can affect recognition of red ('STOP') and blue road signs.
Therefore, I preferred 'global normalization' - removing mean value of the image and dividing by total range.

The main preprocessing step is the image set augmentation. I decided to add variants with distortions that we can
expect in real situation (detecting sing from the moving vehicle) - motion blur and perspective transformation 
(corresponding to the object rotation in a 3D-space).

I decided to:
- balance dataset (equal number of images in each class);
- add some distortions even to the classes with large number of images.

I decided to extend dataset up to 4000 images per class (172000 training images). Only train set was augmented.
In case of cross-validation, augmentation was performed after splitting dataset into train and validation folds.

(*If you want to get a **very good** validation results, you may augment your dataset by a factor of 10 - and then
split it into train and validation set: most of the validation data will be just a variant of training images,
you'll get very good validation accuracy - but poor results for test and for real-world usage of your net.*)


For each new image were chosen randomly:
- motion blur: no motion (60%), moderate blur with 3x3 kernel (30%), strong blur with 5x5 kernel (10%)
- rotation around X (horizontal) axis: [0° 20°] (20° corresponds to the sign seen from below: it makes no sense to use negative angles,
	as normally we do not observe signs from above);
- rotation around Y (vertical) axis: [-35°  35°] (corresponds to the visible rotation of the sign as we pass it);
- rotation around Z (coming 'out of the picture plane') axis: [-15°  15°] (corresponds to the errors in sign placement 
	and to the distortions introduced by wide-angle camera when we pass the sign).




Here is an example of an original image and motion blur distortion applied to that image:

![alt text][image2]

And here you can see rotations applied to the same image

![alt text][image3]

After first experiments with my network, I've added two more distortions: 
- scaling (in range 0.8 - 1.2)

![alt text][image4]

- intensity change (-0.3 +0.3, applied to normalized image with clipping [-0.5 0.5])

![alt text][image5]

After these additions, number of samples per class was extended to 6000 - and up to 10000 for final test.

The sixth code cell of the IPython notebook contains the code for augmenting the data set.


#### 2. Dataset organization. 

In this work I used train, validation and test sets provided in the given files.

In the end, I've performed additional experiment with another approach to the dataset -
I'll mention it in the 'discussion' section.

#### 3. Model architecture.

First preliminary tests were conducted on 'lab' LeNet network.

Then I switched to the variant of the 'Multi-scale convolutional network' by Pierre Sermanet [1][2].
In '*real life* it was the first 'deep' network I used for sign recognition - first using  EBLearn framework by Sermanet [3],
then switching to more flexible (and supported) [Torch](http://torch.ch/) - Lua-based machine learning framework.

Starting from 'original' multi-scale network, some changes were introduced and tested, and
now my implementation has some differences from the original one:
- instead of 5x5 convolutions, I use sequence of two 3x3 convolutions (idea from [4]).
- I use **ReLU** activations between convolutional layers, and **tanh** after the last convolutional layer and between fully-connected layers;
- I use smaller layer sizes to reduce computational speed and to decrease the chance of overfitting.
- I use dropout [5] in fully-connected layer. 

(In 'production' torch code I use [PReLU](https://github.com/torch/nn/blob/master/doc/transfer.md#prelu) activations and [BatchNormalization] layer,
but I'm not ready to implement them in TF)

I've tested several additional models with deeper layers, or additional fully-conneted layer. These models
 clearly overfitted (with zero train loss and validation loss a little bit larger then of chosen network).


The code for my final model is located in the cell **6** of the ipython notebook. 

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
 


#### 4. Model training. 


Models were trained using ADAM optimizer with exponential learning rate decay (starting form 1e-3, it was reduced by a factor of 0.998 after 
100 iterations, so that on 30th epoch effective learning rate was 7.5e-5).
I used batches of size 100.
Weights were initialized with mean 0 and sigma = 0.1 (to get random values near to zero), weights of fully-connected layers were initialized with sigma = 0.01.

During tests, network was trained for 20 epochs, final training consisted of 30 epochs.
With larger number of epochs, model showed signs of overfitting (decrease of the training loss
with non-decreasing validation loss ).

The code for training the model is located in the cells **7**, **8** and **9** of the ipython notebook. 



#### 5. Discussion

The code for calculating the accuracy of the model is located in the cells **8** and **9** of the Ipython notebook.

My final model results were:
* training set accuracy of .998
* validation set accuracy of .982
* test set accuracy of .974

I've planned to use iterative approach - but fished by using first architecture I've build for this project.
It is based on well-knows Pierre Sermaet multi-scale network [1][2], some changes (e.g. usage of ReLU and tanh
activations) I introduced and tested some years ago.

Decision to replace 5x5 convolution by a sequence of two 3x3 convolutions was influenced by [4]. It proved to be right decision: 
I've performed test with '5x5' layer, and it showed the result that was 1% worse.

So, my iteration updates deal not with the network architecture or it's hyper-parameters - 
but with data augmentation:
- rotation increased validation accuracy from .950 to .976
- rotation + motion - to .979
- rotation + motion + intensity - to .981
- rotation + motion + intensity + scaling - to .984


 

### Test a Model on New Images

#### 1. Test on five new images

For this test I used five from more than 80k sign images, collected by [Navmii](http://www.navmii.com) team in 
different European countries:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]



2. Model's predictions on these traffic signs

The code for making predictions on my final model is located in the cells **10 - 13** of the Ipython notebook.


Here are the results of the prediction:

![alt text][image11]

In this table original image is shown in the first column, bar-plot of top-5 
probabilities (in log scale) - in the second column. Last 5 columnts show
the images of top-5 classes.
All signs were predicted correctly, with high probability (probability of the second class
varies from 1e-4 to 1e-8).


But when I took really hard examples from my collection,

![alt text][image12]

this model recognized only seven sign out of ten:

![alt text][image13]

One can notice, that for all three errors top probability is quite low (0.53, 0.33, 0.64). whereas for correct recognition correct class probabilities are greater than 0.9 (from 0.94 to 1.0). It means that we can use that number to measure model's certainly on it's prediction � and discard wrong predictions.

## Additional experiments with dataset.

When the task was completed, I've performed one additional experiment. The idea was,
according to wonderful
Andrew Ng's lecture ['Nuts and bolts of machine learning'](https://www.youtube.com/watch?v=F1ka6a13S9I), 
to use as much of exising data, as we can.
To do it, I:

- merged training and validation data into one 'Dev' set;
- reserved 10% of the set for pre-final test (DevTest set);
- used the rest of the data (DevValTrn set) for 5-fold cross-validation.

Dataset splitting was not completely random: I've preserved the same proportion (10% for DevTest and 20% for Validation)
for each class, merging and shuffling results afterward. Data augmentation was performed on each fold of the
5-fold cross-validation.

Original Test set was left intact and was not used until the last step of the training.

DevTest set contained 3920 images. From the rest Dev set (35289 images), on each fold of the cross-validation 7039 images were
assigned to the Validation set, and 28520 - to the train set, which was augmented to 172000 images (4000 images per class).

In this setup, I got wonderful validation accuracy up to .9985 - but test accuracy was the same.
The explanation is very simple: originally we had different sings (I mean different physical objects) in train and validation sets.
After merging and shuffling, that separation gone away - and I got high score. But this high score
did not reflect real network capability.


### Conclusion

These experiments demonstrate the importance of data for ANN training (and for all
types of ML in general). 


'Hard examples' were successfully classified by a similar network trained with 
the smaller dataset (about 15000 images for 94 sign classes). But those images were carefully selected:
I used only one or two images of the one sign. GTSRB is three times large - but it is highly redundant,
with up to 10 images of the same sign.

I know that data collection (including annotations and <double|tr-ple>-checking) is extremely time-consuming and boring 
task - but it is **very** important for any successful project.






I also want to mention several questions that are (IMHO) still open in this old TSR field:
- **Garbage collector.** My experiments on real-world data showed, that it classification results improve if you add additional class
for different kinds of 'garbage'. Real object ('sign' in our case) detectors are never ideal, normal sign detector
usually loves windows and wheels (and may detect the face of your boss as well). Without 'garbage class', wheel image may obtain some sign label with rather
high probability.  Using 'garbage' class, you can filter out most of the wrong detections.

- **Large number of classes.** I've started sign recognition from the very small set of 'speed limit' signs, then it was extended
to all 'restrictive' sings (round signs with red border). Then I trained separate classifier for warning
(triangular) signs. Both classifiers got high accuracy (up to .985). But when I trained 'united' classifier
for all 92 types, I got only accuracy of 0.96.

     Several methods were proposed as a solution to this problem, including ECOC (error-correction output codes [6],
 wich received direct application to the traffic signs in [7]). I do not thing this problem ceased to exist 
 with successful classification of ImageNet into 1000 classes. May be, you can just skip it - if you have
 millions of points of data (and right answer can be anywhere in top-5 predictions). I think that the most
 interesting direction is the combination of neural networks and trees (or forests) [8][9].

 
- **Transfer leaning**. In many countries, traffic signs are similar (we are not talking about USA) - but they
have some differences: narrow or wide border, white or yellow background. 

    ![alt text][image14] 

    We can suppress color information manually (using grayscale images), and we can get rid
    of the border - but it would be nice to train system on a large number of white German signs -
    and then easily transfer (without additional large-scale training) this knowledge to yellow Finnish signs.




### Literature

1. Sermanet, P. & LeCun, Y. Traffic sign recognition with multi-scale convolutional networks. in Neural Networks (IJCNN), The 2011 International Joint Conference on 2809–2813 (IEEE, 2011).
2. Sermanet, P., Kavukcuoglu, K. & LeCun, Y. Traffic signs and pedestrians vision with multi-scale convolutional networks. in Snowbird Machine Learning Workshop 2, 8 (2011).
3. Sermanet, P., Kavukcuoglu, K. & LeCun, Y. Eblearn: Open-source energy-based learning in c++. in Tools with Artificial Intelligence, 2009. ICTAI’09. 21st International Conference on 693–697 (IEEE, 2009).
4. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. & Wojna, Z. Rethinking the Inception Architecture for Computer Vision. arXiv:1512.00567 [cs] (2015).
5. Srivastava, N., Hinton, G. E., Krizhevsky, A., Sutskever, I. & Salakhutdinov, R. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research 15, 1929�1958 (2014).
6. Dietterich, T. G. & Bakiri, G. Solving multiclass learning problems via error-correcting output codes. arXiv preprint cs/9501101 (1995).
7. Bar?, X., Escalera, S., Vitri?, J., Pujol, O. & Radeva, P. Traffic sign recognition using evolutionary adaboost detection and forest-ECOC classification. Intelligent Transportation Systems, IEEE Transactions on 10, 113�126 (2009).
8. Zhou, Z.-H. & Feng, J. Deep Forest: Towards An Alternative to Deep Neural Networks. arXiv:1702.08835 [cs, stat] (2017).
9. Balestriero, R. Neural Decision Trees. arXiv:1702.07360 [cs, stat] (2017).

