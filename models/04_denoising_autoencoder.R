################################################################################ 
############################ Denoising Autoencoder #############################
################################################################################
require("mxnet")
require("ggplot2")
require("reshape2")

# we do not need class labels
rm(trainLabels)
rm(testLabels)

# set hyperparameter
epochs = 20
batchSize = 64
code = 64 # size (number of neurons) of the code layer of our autoencoder

################################################################################ 
################################## noise data ##################################
################################################################################
# choose noise factor (0.1: almost no noise, 1: alot (!) of noise)
noise_level = 0.6 
trainData_noised = trainData + runif(dim(trainData)[1] * dim(trainData)[2], 
  min = - noise_level, max = noise_level)
trainData_noised[trainData_noised < 0] = 0
trainData_noised[trainData_noised > 1] = 1

testData_noised = testData + runif(dim(testData)[1] * dim(testData)[2], 
  min = - noise_level, max = noise_level)
testData_noised[testData_noised < 0] = 0
testData_noised[testData_noised > 1] = 1

################################################################################ 
############################# plot the noised data #############################
################################################################################
plots = 4
visualize_noise = sample(dim(trainData)[2], plots)
par(mfcol = c(2, plots), cex = 0.1)

# upper plot shows original data, bottom row the noised data we will use to train
for(i in 1:plots){
  
  truth = trainData[1:784, visualize_noise[i]]
  truth_mat = matrix(truth, nrow = 28, ncol = 28, byrow = TRUE)
  truth_mat = apply(truth_mat, 2 , rev)
  image(t(truth_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255)))
  
  noise = trainData_noised[1:784, visualize_noise[i]]
  noise_mat = matrix(noise, nrow = 28, ncol = 28, byrow = TRUE)
  noise_mat = apply(noise_mat, 2 , rev)
  image(t(noise_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255)))
  
}

################################################################################ 
############################# model architecture ###############################
################################################################################
data = mx.symbol.Variable("data")

encoder = mx.symbol.FullyConnected(data, num_hidden = code)
decoder = mx.symbol.FullyConnected(encoder, num_hidden = 784)
activation2 = mx.symbol.Activation(decoder, act_type = "sigmoid")
output = mx.symbol.LinearRegressionOutput(activation2)

mx.set.seed(1337)
logger = mx.metric.logger$new()
devices = mx.cpu()

model = mx.model.FeedForward.create(output,
  X = trainData_noised, y = trainData,
  ctx = mx.cpu(),
  num.round = epochs,
  array.batch.size = batchSize,
  optimizer = "adam",
  initializer = mx.init.uniform(0.01),
  eval.metric = mx.metric.mse,
  epoch.end.callback = mx.callback.log.train.metric(100, logger),
  array.layout = "colmajor")

################################################################################ 
################################ prediction ####################################
################################################################################
pred = predict(model, testData_noised)
dim(pred)

################################################################################ 
############################## plot the results ################################
################################################################################
# just re-perform the following code block if you want to see results for different digits
plots = 4
visualize_pred = sample(dim(testData_noised)[2], plots)
par(mfcol = c(3, plots), cex = 0.02)

for(i in 1:plots){
  
  # truth (test data)
  truth = testData[1:784, visualize_pred[i]]
  truth_mat = matrix(truth, nrow = 28, ncol = 28, byrow = TRUE)
  truth_mat = apply(truth_mat, 2 , rev)
  image(t(truth_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255)))  
    
  # noised data we want to denoise
  noise = testData_noised[1:784, visualize_pred[i]]
  noise_mat = matrix(noise, nrow = 28, ncol = 28, byrow = TRUE)
  noise_mat = apply(noise_mat, 2 , rev)
  image(t(noise_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255))) 
  
  # autoencoder prediction
  autoencoder = pred[1:784, visualize_pred[i]]
  autoencoder_mat = matrix(autoencoder, nrow = 28, ncol = 28, byrow = TRUE)
  autoencoder_mat = apply(autoencoder_mat, 2 , rev)
  image(t(autoencoder_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255)))
  
}

# dev.off()