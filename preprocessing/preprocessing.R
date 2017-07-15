################################################################################ 
################################ Preprocessing #################################
################################################################################ 

# Set your working directory to load the data
setwd("../mxnet-tutorials-in-R/data")

# Load the RData files  
load("Train.RData")
load("Test.RData")

## Training data:
# As you can see, the dimension of the training data are 60000 rows and 784 colums
dim(trainData)
# However, we prefer 784 rows and 60000 colums,  which is a very common convention 
# in many deeplearning frameworks.
trainData = t(trainData)
dim(trainData)

# Currently, the labels are categorical/one-hot encoded.
trainLabels[0:6, 0:9]
# We assign true labels, simply because I don't like one-hot. 
trainLabels_temp = numeric(length = dim(trainLabels)[1])
for(i in 1:9){
  trainLabels_temp = trainLabels_temp + i*trainLabels[, i]
}
rm(i)
trainLabels = trainLabels_temp
rm(trainLabels_temp)
# As we can see, the results match those from the one-hot encoding:
head(trainLabels)

## Test data:
testData = t(testData)
testLabels_temp = numeric(length = dim(testLabels)[1])
for(i in 1:9){
  testLabels_temp = testLabels_temp + i*testLabels[, i]
}
rm(i)
testLabels = testLabels_temp
rm(testLabels_temp)






