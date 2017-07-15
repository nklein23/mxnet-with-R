################################################################################ 
################################# Preparations #################################
################################################################################ 
# We will use the darch package as it gives us an easy access to the mnist data.
# It will download the original files from http://yann.lecun.com/exdb/mnist/ 
# and convert it into RData files, which can be easily read via load().
require("darch")
# Set your working directory
setwd("C:../mxnet-tutorials-in-R/data/")
darch::provideMNIST(folder = "C:/Users/Niklas/mxnet-tutorials-in-R/data/", 
  download = TRUE)


