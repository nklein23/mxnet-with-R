get_dependencies = function(my_dependencies) {

  ## A function to install a collection of packages. Each package will only be installed
  ## if it is not in your package library.
  ##  
  ## Args:
  ##   my_dependencies (vector or list): The packages of interest.
  ##
  ## Returns:
  ##   Two components, a .params file including the weights and a .json containing
  ##   information about the corresponding model architecture.
  ##
  ## Example:
  ##   To download resnet-34, use get_resnet('my_data_folder/', 50)  
  
  # Inspect each component of the input. 
  invisible(lapply(my_dependencies, function(my_package) {
    
    # Check if the component is already installed.
    if (my_package %in% rownames(installed.packages())) {
      
      # Print this.
      cat(paste(my_package), 'is already installed!')
      cat('\n')
      
    # If not, check if the component is MXNet and install the CPU version.
    } else if (my_package == 'mxnet') {
      
      # Print this.
      cat('Installing', paste(my_package), '..')
      
      # Install the actual package.
      cran = getOption('repos')
      cran['dmlc'] = 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/'
      options(repos = cran)
      install.packages('mxnet')  
      
    # Else, install the package.
    } else {
      
      # Print this.
      cat('Installing', paste(my_package), '..')
      cat('\n')
      
      # Install the actual package.
      install.packages(my_package)
      
    }
    
  }))
  
}  
  
  