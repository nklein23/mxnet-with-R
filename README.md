# mxnet examples for people with intermediate R and deeplearning knowledge.
note that I do not intend to find the best model for a given problem, the purpose of this project is simply to give you an easy access to mxnet and its functionality.

## dependencies:

* darch
* mxnet
* ggplot2
* reshape2

## to use the code:

1. run preprocessing/get_mnist.R
2. run preprocessing/conversion.R
3. run models/<any file of your choice>

note that you have to edit the working directory in all files.

## model overview

| Model         | Time to execute*    | Type                                                | Peak accuracy                          |
| ------------- |:-------------------:|:---------------------------------------------------:|:--------------------------------------:|  
| 01            | roughly 1.3 min     | very simple neural network with 3 layers            | 0.9784 @ 50 epochs                     |
| 02            | roughly 43.5 min    | 4 neural nets with dropout (batchsize benchmark)    | best model yields 0.9793 @ 50 epochs   |
| 03            | roughly 13.5 min    | CNN with 3 conv/pooling + 3 dense layers            | 0.9862 accuracy @ 20 epochs            | 
| 04		| roughly 2 min	      | Denoising Autoencoder				    | Judge with your eyes (image below)     |

\* models were executed on mainstream CPU with 4 cores/4 threads @ 3.9 GHZ

I'm planing to add more models in the future (RNN/lstm, image segmentation models)

## model 01 results:

![alt text](https://github.com/NiklasDL/mxnet-tutorials-in-R/blob/master/results/simpleNetErrors.png?raw=true)

## model 02 results:

![alt text](https://github.com/NiklasDL/mxnet-tutorials-in-R/blob/master/results/deepNetTrainError.png?raw=true)

![alt text](https://github.com/NiklasDL/mxnet-tutorials-in-R/blob/master/results/deepNetTestError.png?raw=true)

## model 03 results:

![alt text](https://github.com/NiklasDL/mxnet-tutorials-in-R/blob/master/results/cnnError.png?raw=true)

## model 04 results for arbitrary digits (top row shows original digits, intermediate row noised images used for training and bottom row prediction):

![alt text](https://github.com/NiklasDL/mxnet-tutorials-in-R/blob/master/results/denoising_autoencoder.png?raw=true)