########################################
######## Download the MNIST data #######
########################################

# Load the get_mnist function.
get_mnist_fun = paste(dirname(getwd()), 'utils', 'get_mnist.R', sep = '/')
source(get_mnist_fun)

# Define directory for the data.
my_data_dir = paste(dirname(getwd()), 'data/', sep = '/')

# Download MNIST.
get_mnist(my_data_dir)


########################################
####### Preprocess the MNIST data ######
########################################

# Load the preprocessing function.
preprocessing_fun = paste(dirname(getwd()), 'utils', 'preprocessing.R', sep = '/')
source(preprocessing_fun)

# Create categorical labels for the train data.
load(paste(my_data_dir, 'train.RData', sep = ''))
trainLabels = apply(trainLabels, 1, function(x) to_categorical(x))

# Create categorical labels for the test data.
load(paste(my_data_dir, 'test.RData', sep = ''))
testLabels = apply(testLabels, 1, function(x) to_categorical(x))


########################################
### Set a couple of mxnet parameters ###
########################################

# Load the package.
require('mxnet')

# Define the device, either mx.cpu() or mx.gpu().
devices = mx.cpu()

# We want a seed for reproducible results.
mx.set.seed(1)

# MXNet uses logger objects to save results during training.
logger = mx.metric.logger$new()


########################################
###### Build dense feedforward net #####
########################################

# Create the model's input layer.
data = mx.symbol.Variable('data')

# The first layer has 32 units and relu activation.
fc1 = mx.symbol.FullyConnected(data, name = 'fc1', num_hidden = 32)
act1 = mx.symbol.Activation(fc1, name = 'relu1', act_type = 'relu')

# The second layer has 64 units and relu activation.
fc2 = mx.symbol.FullyConnected(act1, name = 'fc3', num_hidden = 64)
act2 = mx.symbol.Activation(fc2, name = 'relu3', act_type = 'relu')

# The output layer has 10 units and softmax activation.
fc3 = mx.symbol.FullyConnected(act2, name = 'fc4', num_hidden = 10)
softmax = mx.symbol.SoftmaxOutput(fc3, name = 'sm')


########################################
#### Define model's hyperparameters ####
########################################

# MXNet follows a 'colmajor philosophy' (for whatever reason), thus it 
# is  recommended to transpose the data before feeding it into the model.
trainData = t(trainData)
testData = t(testData)

# Set the number of epochs we want to train the model .
num_epochs = 5
my_batchsize = 32

# Create a data frame for the results.
results = data.frame(matrix(ncol = 2, nrow = num_epochs))

# Train the model.
model = mx.model.FeedForward.create(softmax,
  X = trainData, y = trainLabels,
  eval.data = list(data = testData, label = testLabels),
  ctx = devices,
  optimizer = 'sgd',
  learning.rate = 0.001,
  momentum = 0.9,
  wd = 0.001,
  num.round = num_epochs,
  array.batch.size = my_batchsize,
  initializer = mx.init.uniform(0.03),
  eval.metric = mx.metric.accuracy,
  epoch.end.callback = mx.callback.log.train.metric(5, logger))
    
  
########################################
####### Some fancy visualizations ######
########################################

# Fill the results data frame with the.
results[1] = as.numeric(lapply(logger$train, function(x) 1 - x))
colnames(results)[1] = paste("Train")

results[2] = as.numeric(lapply(logger$eval, function(x) 1 - x))
colnames(results)[2] = paste("Test")  
  
# Load and call the visualization function.
get_vis_fun = paste(dirname(getwd()), 'utils', 'visualize_results.R', sep = '/')
source(get_vis_fun)
visualize_results(results, 'Misclassification rate', 1)
