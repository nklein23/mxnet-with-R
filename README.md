# mxnet examples for people with intermediate R and deeplearning knowledge.

## dependencies:

* darch
* mxnet
* ggplot2
* reshape2

## to use the code:

1. run preprocessing/get_mnist.R
2. run preprocessing/conversion.R
3. run models/<any file of your choice>

### You have to edit the working directory in all files.


| Model         | Time to execute*    | Type                                                |
| ------------- |:-------------------:|:---------------------------------------------------:|  
| 01            | roughly 1.3 min     | very simple neural network with 3 layers            |
| 02            | roughly 43.5 min    | 4 neural nets with 4 layers + dropout (benchmark)   |
| 03            | up to come          | 3 layer CNN                                         |

\* models were executed on mainstream CPU with 4 cores/4 threads @ 3.9 GHZ

![alt text](https://github.com/NiklasDL/mxnet-tutorials-in-R/blob/master/results/simpleNetErrors.png?raw=true)

![alt text](https://github.com/NiklasDL/mxnet-tutorials-in-R/blob/master/results/deepNetTrainError.png?raw=true)

![alt text](https://github.com/NiklasDL/mxnet-tutorials-in-R/blob/master/results/deepNetTestError.png?raw=true)