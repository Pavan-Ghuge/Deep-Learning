# Deep-Learning


# Assignment 1:
Write a CUDA program for computing the dot product of a vector in parallel with 
each row of a matrix. You are required to have each thread access consecutive
memory locations (coalescent memory access). The inputs are 

1. number of rows
2. number of columns
3. a data matrix file similar to the format in the Chi2 program 
4. a vector file (one row)
5. cuda device
6. number of threads

For example if the input is

1 2 0
1 1 0
1 2 1

and w = (2, 4, 6)

then your program should output

10
6
16

Compute the dot products in parallel your kernel function. You will have to
transpose the data matrix in order to get coalescent memory access. 


# Assignment 2:

Convert the CUDA program that you wrote for assignment one into an
OpenMP one. The output of both your CUDA and OpenMP programs must be the same. 

In order to use openmp on Lochness you must type

module load intel/compiler/2017.2.174

When submtting a job to the cluster you have to specify the number of cores
that you need. Type 

sbatch slurmscript


# Assignment 3:

Write a Python program that trains a single layer neural network
with sigmoid activation. You may use numpy. Your input is in dense 
liblinear format which means you exclude the dimension and include 0's. 

Let your program command line be:

python single_layer_nn.py <train> <test> <n>

where n is the number of nodes in the single hidden layer.

For this assignment you basically have to implement gradient
descent. Use the update equations we derived on our google document
shared with the class.

Test your program on the XOR dataset:

1 0 0
1 1 1
-1 0 1
-1 1 0

1. Does your network reach 0 training error? 

2. Can you make your program into stochastic gradient descent (SGD)?

3. Does SGD give lower test error than full gradient descent?

4. What happens if change the activation to sign? Will the same algorithm
work? If not what will you change to make the algorithm converge to a local
minimum?


# Assignment 4:

Implement stochastic gradient descent (SGD) in your back propagation program
that you wrote in assignment 3. In the original SGD algorithm we update the 
gradient based on a single datapoint:

SGD algorithm:

Initialize random weights
for(k = 0 to n_epochs):
	Shuffle the rows (or row indices)
	for j = 0 to rows-1:
		Determine gradient using just the jth datapoint
		Update weights with gradient
	Recalculate objective

We will modify this into the mini-batch version and implement it for this
assignment.

I. Mini-batch SGD algorithm:

Initialize random weights
for(k = 0 to n_epochs):
	for j = 0 to rows-1:
		Shuffle the rows (or row indices)
		Select the first k datapoints where k is the mini-batch size
		Determine gradient using just the selected k datapoints
		Update weights with gradient
	Recalculate objective

Your input, output, and command line parameters are the same as assignment 3.
We take the batch size k as input. We leave the offset for the final layer 
to be zero at this time.

Test your program on the XOR dataset:

1 0 0
1 1 1
-1 0 1
-1 1 0

1. Test your program on breast cancer and ionosphere given on the website. Is the 
mini-batch faster or the original one? How about accuracy?

2. Is the search faster or more accurate if you keep track of the best objective
in the inner loop?


# Assignment 5: 
Write a Python program that trains a neural network with a single 2x2
convolutional layer with stride 1 and global average pooling. See
our course notes on google drive for equation updates with sigmoid
activation. 

The input are 3x3 images. Images for training are going to be in
one directory called train and test ones in the directory called
test. The train directory has a csv file called data.csv that contains
the name of each image dataset and its label. For example your data.csv
would look like

Name,Label
image0.txt,1
image1.txt,0

where image0.txt is 

1 0 0
0 1 0
0 0 1

and image1.txt is 

0 0 1
0 1 0
1 0 0

Let your program command line be:

python convnet.py <train> <test> 

1. What is the convolutional kernel learnt by your program? 


# Assignment 6:
Write a convolutional network in Keras to train the Mini-ImageNet 
dataset on the course website. Your constraint is to create a network
that achieves at least 80% test accuracy (in order to get full points).

Submit your assignments as two files train.py and test.py. Make
train.py take three inputs: the input training data, training labels,
and a model file name to save the model to. 

python train.py <train.npy> <trainlablels.npy> <model file>

It is straightforward to save a Keras model to file, see the simple example here 
https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state

Make test.py take three inputs: the input test data, test labels,
and a model file name to load the model. 

python test.py <test.npy> <testlabels.npy> <model file>

The output of test.py is the test error of the data which is
the number of misclassifications divided by size of the test set.

# Assignment 7:

Write a convolutional network in Keras to train the Mini-ImageNet 
dataset on the course website. You may use transfer learning. Your
goal is to achieve above 90% accuracy on the test/validation datasets.

Submit your assignments as two files train.py and test.py. Make
train.py take two inputs: the input training directory
and a model file name to save the model to.

python train.py train <model file>

Make test.py take two inputs: the test directory
and a model file name to load the model.

python test.py test <model file>

The output of test.py is the test error of the data which is
the number of misclassifications divided by size of the test set.

In addition to your transfer learning solution also submit a
solution without transfer learning. In other words what is the
maximum test accuracy that you can obtain with a custom designed
model? Submit this as train2.py and test2.py with the same
parameters as above.

# Assignment 8:

Classify images in the three Kaggle datasets on the course website 
with convolutional networks. You may use transfer learning. Your
goal is to achieve above 90% accuracy on the test/validation datasets.

You may access the datasets directly from my home directory on
lochness:

/home/u/usman/cs_677_datasets

Submit your assignments as two files train.py and test.py. Make
train.py take two inputs: the input training directory
and a model file name to save the model to. 

python train.py train <model file>

Make test.py take three inputs: the test directory 
and a model file name to load the model. 

python test.py test <model file>

The output of test.py is the test error of the data which is
the number of misclassifications divided by size of the test set.


# Assignment 9:


Implement a simple GAN in Keras to generate MNIST images. Use the GAN given here

https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f

as your discriminator and generator. 

You want to train the generator to produce images of numbers between 0 and 9.

Submit your assignments as two files train.py and test.py. Make
train.py take two inputs: the input training directory
and a model file name to save the generator model to. 

python train.py MNIST_train_directory <generator model file>

Make test.py take one input: the generator model file. The output
of test.py should be images resembling MNIST digits saved to the output
file.

python test.py <generator model file> <output image filename>
  
  
  
 # Assignment 10:
 estLabels = np.load(testLabelsFile)

testData = testData[np.logical_or(testLabels == 0, testLabels == 1)]
testLabels = testLabels[np.logical_or(testLabels == 0, testLabels == 1)]
testLabels = keras.utils.to_categorical(testLabels, 2)

We normalize each image by subtracting the mean:

testDataMean = np.mean(valData, axis=0)
testData = valData - valDataMean

A successful attack should have a classification accuracy of at most 10%
on the test.

Submit your assignments as a single file wbattack.py. Make
train.py take three inputs: the test data and labels and the target model to 
attack (in our case this is the network with 20 hidden nodes).

python wbattack.py <test data> <test labels> <target model to be attacked> 

Your wbattack.py program should create adversaries x' for every image 
in the test set using the formula x' = x + epsilon*sign(grad_x(f(x,y))) where
epsilon=0.0625, grad_x(f(x,y)) is the gradient of the model f(x,y) with respect
to the training data x, and x and y are the training data and training labels
repsectively. We can obtain the gradient of the model w.r.t the training data
in Keras (see code below).

from keras import backend as K

gradients = K.gradients(model.output, model.input)[0]
iterate = K.function(model.input, gradients)
grad = iterate([traindata])
evaluated_gradient = grad[0]  

Instead of the last two lines above you may also use

gradients = calculategrads(traindata)[0]

After creating adversaries evaluate their output from the target model f(x,y).
A successful white box attack should have adversarial accuracy (which is the
accuracy of adversarial examples) below 10%. 


# Assignment 11:

Learn a word2vec model from fake news dataset and a real news dataset. We 
will use the word2vec model implemented in the Python Gensim library. Use
a hidden layer of 300 and 1000 nodes separately to represent each word 
in the word2vec training. Use two different window sizes of 2 and 5. 

You can set workers to 10 and train it faster on multiple cores. In order to
do this you have to set tasks-per-node to 10 like below:

#SBATCH --tasks-per-node=10

Now we have two sets of word representations learnt from different datasets. 
Output the top 5 most similar words to the following ones from each 
representation.

1. Hillary
2. Trump
3. Obama
4. Immigration

In order to do this we first normalize all vector representations (set them 
to Euclidean length 1). Consider the vector x for a given word w. We 
compare the cosine similarity between x and the vectors x' for each word w' 
in the fake news dataset first. We then output the top 5 words with highest 
similarity. We then do the same for the real news and then see if the top 
similar words differ considerably.

Submit your assignments as two files train.py and test.py. Make
train.py take two inputs: the text dataset on which to learn the words and
a model file name to save the word2vec model to.

python train.py <text data> <word2vec model file to save the model> 

Make test.py take three inputs: text dataset, word2vec model, a query file 
containing five query words. The output should be the top five most similar 
words to each word in the query file.

python test.py <text data> <word2vec model file from train.py> <query words filename>

Are the most similar words to the queries considerably different from the 
fake and real news datasets? 
