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
| 01            | roughly 1.3 min     | single neural net with 3 layers                     |
| 02            | up to come          | 4 neural nets with 3 layers (batch size benchmark)  |
| 03            | up to come          | 3 layer CNN                                         |

* models were executed on mainstream CPU with 4 cores/4 threads @ 3.9 GHZ