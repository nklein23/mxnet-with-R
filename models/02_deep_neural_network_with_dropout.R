################################################################################ 
#################### Deep Neural Network with Dropout ##########################
################################################################################

# Set your working directory to save the results from the computations
setwd("../mxnet-tutorials-in-R/results")

require("mxnet")
require("ggplot2")
require("reshape2")

epochs = 25
batchSize = c(10, 50, 100, 200)

resultsTrain = data.frame(matrix(ncol = length(batchSize),
  nrow = epochs))
resultsTest = data.frame(matrix(ncol = length(batchSize), 
  nrow = epochs))

for(i in seq_along(batchSize)){
  
  print(paste0("Batchsize iteration ", i, " of ", length(batchSize)))
  
  data = mx.symbol.Variable("data")
  drop0 = mx.symbol.Dropout(data = data, p = 0.4)
  fc1 = mx.symbol.FullyConnected(drop0, name = "fc1", num_hidden = 256)
  act1 = mx.symbol.Activation(fc1, name = "relu1", act_type = "relu")
  drop1 = mx.symbol.Dropout(data = act1, p = 0.2)
  fc2 = mx.symbol.FullyConnected(drop1, name = "fc2", num_hidden = 128)
  act2 = mx.symbol.Activation(fc2, name = "relu2", act_type = "relu")
  drop2 = mx.symbol.Dropout(act2, p = 0.2)
  fc3 = mx.symbol.FullyConnected(act2, name = "fc3", num_hidden = 64)
  act3 = mx.symbol.Activation(fc3, name = "relu3", act_type = "relu")
  drop3 = mx.symbol.Dropout(act3, p = 0.2)
  fc4 = mx.symbol.FullyConnected(act3, name = "fc4", num_hidden = 10)
  softmax = mx.symbol.SoftmaxOutput(fc4, name = "sm")
  
  devices = mx.cpu()
  mx.set.seed(1337)
  logger = mx.metric.logger$new()
  
  model = mx.model.FeedForward.create(softmax,
    X = trainData, y = trainLabels,
    eval.data = list(data = testData, label = testLabels),
    ctx = devices,
    optimizer = "sgd",
    learning.rate = 0.03,
    momentum = 0.9,
    wd = 0.001,
    num.round = epochs,
    array.batch.size = batchSize[i],
    eval.metric = mx.metric.accuracy,
    initializer = mx.init.uniform(0.07),
    epoch.end.callback = mx.callback.log.train.metric(5, logger))
  
  resultsTrain[i] = as.numeric(lapply(logger$train, function(x) 1-x))
  colnames(resultsTrain)[i] = paste("Batchsize ", 
    batchSize[i], sep = "")
  
  resultsTest[i] = as.numeric(lapply(logger$eval, function(x) 1-x))
  colnames(resultsTest)[i] = paste("Batchsize ", 
    batchSize[i], sep="")
  
}
################################################################################ 
############################ Plot Training Errors ##############################
################################################################################ 

nnTrainError = as.data.frame(cbind(1:dim(resultsTrain)[1], resultsTrain))
colnames(nnTrainError)[1] = "epoch"
nnTrainError = melt(nnTrainError, id = "epoch")

write.csv(nnTrainError, file = "nnTrainError", row.names = FALSE, quote = FALSE)

nnTrainError = read.csv("nnTrainError", header = TRUE)
options(scipen=999)
nnTrainError$variable = factor(nnTrainError$variable)

ggplot(data = nnTrainError, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "train error", limits = c(0, 1)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") + 
  labs(colour = "Batchsize")

################################################################################ 
############################## Plot Test## Errors ##############################
################################################################################ 

nnTestError = as.data.frame(cbind(1:dim(resultsTest)[1], resultsTest))
colnames(nnTestError)[1] = "epoch"
nnTestError = melt(nnTestError, id = "epoch")

write.csv(nnTestError, file = "nnTestError", row.names = FALSE, quote = FALSE)

nnTestError = read.csv("nnTestError", header = TRUE)
options(scipen=999)
nnTestError$variable = factor(nnTestError$variable)

ggplot(data = nnTestError, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "test error", limits = c(0, 0.5)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") + 
  labs(colour = "Batchsize")

