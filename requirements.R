########################################
###### Install necessary packages ######
########################################

# Do you own a Nvidia GPU and have the CUDA toolkit installed? 
# Then go to https://mxnet.incubator.apache.org/install/index.html
# and install MXNet GPU implementation accordingly. 
# This Script will only install the CPU version.

# Load the get_mnist function.
get_dependencies = file.path(dirname(getwd()), 'mxnet-with-r/utils', 'get_dependencies.R')
source(get_dependencies)

# We need the following packages:
my_dependencies = c('mxnet', 'devtools', 'darch', 'ggplot2', 'reshape2', 'stringr')

# Each package will only be installed if it is not in your package library.
get_dependencies(my_dependencies)

