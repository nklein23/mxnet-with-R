########################################
######## Download the MNIST data #######
########################################

# Load the get_mnist function.
get_data_mnist = file.path(dirname(getwd()), 'utils', 'get_data_mnist.R')
source(get_data_mnist)

# Define directory for the data.
my_data_dir = file.path(dirname(getwd()), 'data')

# Download MNIST.
get_data_mnist(my_data_dir)


########################################
####### Preprocess the MNIST data ######
########################################

# Load the preprocessing function.
to_categorical = file.path(dirname(getwd()), 'utils', 'to_categorical.R')
source(to_categorical)

# Create categorical labels for the train data.
load(file.path(my_data_dir, 'train.RData'))
trainLabels = apply(trainLabels, 1, function(x) to_categorical(x))

# Create categorical labels for the test data.
load(file.path(my_data_dir, 'test.RData'))
testLabels = apply(testLabels, 1, function(x) to_categorical(x))


########################################
### Set a couple of mxnet parameters ###
########################################

# Load the package.
require('mxnet')

# Define the device, either mx.cpu() or mx.gpu().
devices = mx.gpu()

# We want a seed for reproducible results.
mx.set.seed(1)

# MXNet uses logger objects to save results during training.
logger = mx.metric.logger$new()


########################################
###### Build dense feedforward net #####
########################################

# Create the model's input layer.
input = mx.symbol.Variable('data')

# The first convolutional layer uses 3x3 filters, relu activation and max pooling.
conv1 = mx.symbol.Convolution(data = input, kernel = c(3, 3), num_filter = 16)
act1 = mx.symbol.Activation(data = conv1, act_type = 'relu')
pool1 = mx.symbol.Pooling(data = act1, pool_type = 'max', kernel = c(2, 2), stride = c(2, 2))

# The second convolutional layer uses 3x3 filters, relu activation and max pooling.
conv2 = mx.symbol.Convolution(data = pool1,  kernel = c(3, 3), num_filter = 32)
act2 = mx.symbol.Activation(data = conv2, act_type = 'relu')
pool2 = mx.symbol.Pooling(data = act2, pool_type = 'max', kernel = c(2, 2), stride = c(2, 2))

# A flatten layer.
flatten = mx.symbol.Flatten(data = pool2)

# The first dense layer has 8 units and relu activation.
fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 8)
act4 = mx.symbol.Activation(data = fc1, act_type = 'relu')

# The secomd dense layer has 16 units and relu activation.
fc2 = mx.symbol.FullyConnected(data = act4, num_hidden = 16)
act5 = mx.symbol.Activation(data = fc2, act_type = 'relu')

# The output layer has 10 units and softmax activation.
fc3 = mx.symbol.FullyConnected(data = act5, num_hidden = 10)
softmax = mx.symbol.SoftmaxOutput(data = fc3)


########################################
#### Define model's hyperparameters ####
########################################

# MXNet follows a 'colmajor philosophy' (for whatever reason), thus it 
# is  recommended to transpose the data before feeding it into the model.
trainData = t(trainData)
testData = t(testData)

# Reshape the data to fit into the cnn.
dim(trainData) = c(28, 28, 1, ncol(trainData))
dim(testData) = c(28, 28, 1, ncol(testData))

# Set the number of epochs we want to train the model.
num_epochs = 5
my_batchsize = 32

# Create a data frame for the results.
results = data.frame(matrix(ncol = 2, nrow = num_epochs))

# Train the model.
model = mx.model.FeedForward.create(softmax,
  X = trainData, y = trainLabels,
  eval.data = list(data = testData, label = testLabels),
  ctx = devices,
  optimizer = 'adam',
  learning.rate = 0.001,
  num.round = num_epochs,
  array.batch.size = my_batchsize,
  initializer = mx.init.uniform(0.03),
  eval.metric = mx.metric.accuracy,
  epoch.end.callback = mx.callback.log.train.metric(5, logger))
      
  
########################################
####### Some fancy visualizations ######
########################################

# Extract the training results.
results[1] = as.numeric(lapply(logger$train, function(x) 1 - x))
colnames(results)[1] = paste('Train')

results[2] = as.numeric(lapply(logger$eval, function(x) 1 - x))
colnames(results)[2] = paste('Test')  
  
# Load and call the visualization function.
vis_results = file.path(dirname(getwd()), 'utils', 'vis_results.R')
source(vis_results)
vis_results(my_training_results = results, 
            custom_string = 'Misclassification rate', 
            my_ylim = 0.4)

