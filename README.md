# mxnet examples in R 
the purpose of this repo is simply to give you an easy access to mxnet and its api

## dependencies:

* mxnet
* ggplot2
* reshape2
* darch

## to use the code:

1. run preprocessing/get_mnist.R
2. run preprocessing/conversion.R
3. run models/

note that you have to edit directories in some files (to load data or save results)

## model overview

| Model         | Time to execute*    | Type                                                |
| ------------- |:-------------------:|:---------------------------------------------------:|
| 01            | roughly 1.3 min     | very simple (dense) network with 3 layers           |
| 02            | roughly 43.5 min    | 4 dense networks with dropout (batchsize benchmark  |
| 03            | roughly 13.5 min    | CNN with 3 conv + 3 dense layers                    | 
| 04		| roughly 2 min	      | Denoising Autoencoder				    |

\* models were executed on mainstream CPU with 4 cores/4 threads @ 3.9 GHZ

I'm planing to add more models in the future (RNNs/lstm, image segmentation models)

## model 01 results:

![alt text](https://github.com/NiklasDL/mxnet-tutorials-in-R/blob/master/results/simpleNetErrors.png?raw=true)

## model 02 results:

![alt text](https://github.com/NiklasDL/mxnet-tutorials-in-R/blob/master/results/deepNetTrainError.png?raw=true)

![alt text](https://github.com/NiklasDL/mxnet-tutorials-in-R/blob/master/results/deepNetTestError.png?raw=true)

## model 03 results:

![alt text](https://github.com/NiklasDL/mxnet-tutorials-in-R/blob/master/results/cnnError.png?raw=true)

## model 04 results for arbitrary digits (top row shows original digits, intermediate row noised images used for training and bottom row prediction):

![alt text](https://github.com/NiklasDL/mxnet-tutorials-in-R/blob/master/results/denoising_autoencoder.png?raw=true)