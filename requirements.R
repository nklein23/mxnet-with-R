########################################
###### Install necessary packages ######
########################################

# Do you own a Nvidia GPU and have the CUDA toolkit installed? 
# Then go to https://mxnet.incubator.apache.org/install/index.html
# and install MXNet GPU implementation accordingly. 
# This Script will only install the CPU version.

# Check if mxnet is installed..
if (!'mxnet' %in% rownames(installed.packages())) {
  print('installing mxnet..')
  cran = getOption('repos')
  cran['dmlc'] = 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/'
  options(repos = cran)
  install.packages('mxnet')
} else {
  print('mxnet is already installed!')
}

# Check if devtools is installed..
if (!'devtools' %in% rownames(installed.packages())) {
  print('installing devtools..')
  install.packages('devtools')
} else {
  print('devtools is already installed!')
}

# Check if darch is installed..
if (!'darch' %in% rownames(installed.packages())) {
  print('installing darch..')
  devtools::install_github('maddin79/darch')
} else {
  print('darch is already installed!')
}

# Check if ggplot2 is installed..
if (!'ggplot2' %in% rownames(installed.packages())) {
  print('installing ggplot2..')
  install.packages('ggplot2')
} else {
  print('ggplot2 is already installed!')
}

# Check if reshape2 is installed..
if (!'reshape2' %in% rownames(installed.packages())) {
  print('installing reshape2..')
  install.packages('reshape2')
} else {
  print('reshape2 is already installed!')
}
