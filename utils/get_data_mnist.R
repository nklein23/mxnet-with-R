get_data_mnist = function(my_data_dir) {
  
  ## A function to download the original MNIST data from http://yann.lecun.com/exdb/mnist/.
  ## It converts it to .RData files and deletes the compressed archives.
  ##  
  ## Args:
  ##   my_data_dir (string): The path to the folder in which the data is to be stored.
  ##
  ## Returns:
  ##   Two .RData files, the original train and test split.
  ##
  ## Example:
  ##   get_mnist('my_data_folder/')
  
  # Check if the data is incomplete or does not exist.
  
  if (!file.exists(file.path(my_data_dir, 'train.RData')) |
      !file.exists(file.path(my_data_dir, 'testn.RData'))) {
    
    # Download the actual MNIST data.
    require('darch')
    print('Downloading MNIST..')
    suppressWarnings(provideMNIST(folder = paste0(my_data_dir, '/'), download = TRUE))
    
    # Delete unnecessary compressed files.
    invisible(file.remove(list = c(file.path(my_data_dir, 'train-images-idx3-ubyte.gz'),
                                   file.path(my_data_dir, 'train-labels-idx1-ubyte.gz'),
                                   file.path(my_data_dir, 't10k-images-idx3-ubyte.gz'),
                                   file.path(my_data_dir, 't10k-labels-idx1-ubyte.gz'))))
    
  } else {
    
    # Print this for reasons.
    print('The MNIST data are already in your folder.')
    
  }
  
}

