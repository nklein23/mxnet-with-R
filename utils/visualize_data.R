vis_random_data = function(my_data, num_plots, my_seed) {
  
  ## A function to visualize random images from the MNIST data.
  ##  
  ## Args:
  ##   my_data (data.frame): The raw data.
  ##   num_plots (int): The number of images you would like to plot.
  ##   my_seed (int): A seed for reproducible plots.
  ##
  ## Returns:
  ##   |num_plots| plots.
  ##
  ## Example:
  ##   vis_random_data(my_data, 81)
  
  # Set the seed. 
  set.seed(my_seed)
  
  # Transpose the data.
  my_data = t(my_data)
  
  # Sample random indices to plot.
  vis_indices = sample(dim(my_data)[2], num_plots)
  
  # Create a grid to place the plots.
  par(mfcol = c(ceiling(sqrt(num_plots)), ceiling(sqrt(num_plots))), cex = 0.02)
  
  # Plot one tile for each integer.
  for(i in 1:num_plots) {
    truth = my_data[1:784, vis_indices[i]]
    truth_mat = matrix(truth, nrow = 28, ncol = 28, byrow = TRUE)
    truth_mat = apply(truth_mat, 2 , rev)
    image(t(truth_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255)))
  }
}
