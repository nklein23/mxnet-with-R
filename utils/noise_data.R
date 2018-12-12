noise_data = function(my_data, noise_factor) {
  
  ## A function to put random noise on the MNIST data.
  ##  
  ## Args:
  ##   my_data (data.frame): The raw data.
  ##   noise_factor (int): A float between 0 and 1.
  ##
  ## Returns:
  ##   my_data_noised (data.frame): The noised data.
  ##
  ## Example:
  ##   noise_data(my_data, 0.5)
  
  # Sample from a uniform distribution to noise the data. 
  my_data_noised = my_data + runif(dim(my_data)[1] * dim(my_data)[2],
                                   min = - noise_factor, max = noise_factor)
  
  # Lower and upper limits for the normalized pixels.
  my_data_noised[my_data_noised < 0] = 0
  my_data_noised[my_data_noised > 1] = 1
  
  # Return this.
  return(my_data_noised)

}
