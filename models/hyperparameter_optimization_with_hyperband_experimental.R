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

# Create problem
problem = list(train_data = trainData, train_labels = trainLabels,
               test_data = testData, test_labels = testLabels)


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

library('ParamHelpers')
library('hyperbandr')

# hyperparam space
configSpace = makeParamSet(
  makeNumericParam(id = 'input_dropout', lower = 0, upper = 0.6),
  makeNumericParam(id = 'layer1_dropout', lower = 0, upper = 0.2),
  makeNumericParam(id = 'layer2_dropout', lower = 0, upper = 0.2),
  makeNumericParam(id = 'momentum', lower = 0, upper = 0.99)
)

sample.fun = function(par.set, n.configs, ...) {
  # sample from the par.set and remove all NAs
  lapply(sampleValues(par = par.set, n = n.configs), function(x) x[!is.na(x)])
}

my_sample = sample.fun(configSpace, 1)
my_sample[[1]]$input_dropout

my_test_sample = my_sample[[1]]
my_test_sample

######################
input_dropout = 0.5
layer1_dropout = 0.5
layer2_dropout = 0.5

init.fun = function(r, config, problem) {

  ########################################
  ###### Build dense feedforward net #####
  ########################################
  
  # Get Hyperparameters
  # input_dropout = config$input_dropout
  # layer1_dropout = config$layer1_dropout
  # layer2_dropout = config$layer2_dropout
  # my_momentum = config$momentum
  input_dropout = 0.2
  layer1_dropout = 0.2
  layer2_dropout = 0.2
  my_momentum = 0.9
  
  # Input layer.
  data = mx.symbol.Variable('data')
  drop0 = mx.symbol.Dropout(data = data, p = input_dropout)
  # First hidden layer.
  fc1 = mx.symbol.FullyConnected(drop0, name = 'fc1', num_hidden = 32)
  act1 = mx.symbol.Activation(fc1, name = 'relu1', act_type = 'relu')
  drop1 = mx.symbol.Dropout(data = act1, p = layer1_dropout)
  # Second hidden layer.
  fc2 = mx.symbol.FullyConnected(drop1, name = 'fc2', num_hidden = 64)
  act2 = mx.symbol.Activation(fc2, name = 'relu2', act_type = 'relu')
  drop2 = mx.symbol.Dropout(data = act2, p = layer2_dropout)
  # Output layer.
  fc3 = mx.symbol.FullyConnected(drop2, name = 'fc3', num_hidden = 10)
  softmax = mx.symbol.SoftmaxOutput(fc3, name = 'sm')

  ########################################
  ########### Train the model ############
  ########################################  
  
  model = mx.model.FeedForward.create(softmax,
    X = problem$train_data, y = problem$train_labels,
    ctx = devices,
    optimizer = 'sgd',
    learning.rate = 0.001,
    momentum = my_momentum,
    wd = 0.001,
    num.round = r,
    array.batch.size = my_batchsize,
    initializer = mx.init.uniform(0.03),
    eval.metric = mx.metric.accuracy,
    epoch.end.callback = mx.callback.log.train.metric(5, logger))
  
  # Return the model.
  return(model)

}

my_mod = init.fun(r = 5, my_test_sample, problem)



