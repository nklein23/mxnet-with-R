################################################################################ 
############################### Neural Network #################################
################################################################################

# Set your working directory to save the results from the computations
setwd("../mxnet-tutorials-in-R/results")

require("mxnet")
require("ggplot2")
require("reshape2")

epochs = 50
batchSize = 100

results = data.frame(matrix(ncol = 2, nrow = epochs))

data = mx.symbol.Variable("data")
fc1 = mx.symbol.FullyConnected(data, name = "fc1", num_hidden = 128)
act1 = mx.symbol.Activation(fc1, name = "relu1", act_type = "relu")
fc2 = mx.symbol.FullyConnected(act1, name = "fc3", num_hidden = 64)
act2 = mx.symbol.Activation(fc2, name = "relu3", act_type = "relu")
fc3 = mx.symbol.FullyConnected(act2, name = "fc4", num_hidden = 10)
softmax = mx.symbol.SoftmaxOutput(fc3, name = "sm")
  
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
  array.batch.size = 100,
  eval.metric = mx.metric.accuracy,
  initializer = mx.init.uniform(0.07),
  epoch.end.callback = mx.callback.log.train.metric(5, logger))

results[1] = as.numeric(lapply(logger$train, function(x) 1-x))
colnames(results)[1] = paste("Training Error")

results[2] = as.numeric(lapply(logger$eval, function(x) 1-x))
colnames(results)[2] = paste("Test Error")
  
################################################################################ 
################################# Plot Errors ##################################
################################################################################ 

simpleNetError = as.data.frame(cbind(1:dim(results)[1], results))
colnames(simpleNetError)[1] = "epoch"
simpleNetError = melt(simpleNetError, id = "epoch")

write.csv(simpleNetError, file = "simpleNetError", row.names = FALSE, quote = FALSE)

simpleNetError = read.csv("simpleNetError", header = TRUE)
options(scipen=999)
simpleNetError$variable = factor(simpleNetError$variable)

ggplot(data = simpleNetError, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "misclassification", limits = c(0, 0.2)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") + 
  labs(colour = "")

ggsave("simpleNetErrors", device = "png")
