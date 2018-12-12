get_mnist = function(my_data_dir) {
  
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
  if (!file.exists(paste(my_data_dir, 'train.RData', sep = '')) |
      !file.exists(paste(my_data_dir, 'test.RData', sep = ''))) {
    
    # Clear the folder.
    unlink(paste(my_data_dir, '*', sep = ''))
    
    # Download the actual MNIST data.
    require('darch')
    print('Downloading MNIST..')
    darch::provideMNIST(folder = my_data_dir, download = TRUE)
    
    # Delete unnecessary compressed files.
    file.remove(list = c(paste(my_data_dir, 'train-images-idx3-ubyte.gz', sep = ''), 
                         paste(my_data_dir, 'train-labels-idx1-ubyte.gz', sep = ''),
                         paste(my_data_dir, 't10k-images-idx3-ubyte.gz', sep = ''),
                         paste(my_data_dir, 't10k-labels-idx1-ubyte.gz', sep = '')))
    
  } else {
    
    # Print this for reasons.
    print('The MNIST data are already in your folder.')
    
  }
}
