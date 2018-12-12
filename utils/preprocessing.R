to_categorical = function(my_data_point) {
  
  ## A function to convert one-hot encoded labels to categorical labels.
  ##  
  ## Args:
  ##   my_data_point (num): Numeric vector containing one-hot encoded labels.
  ##
  ## Returns:
  ##   my_data_point (int): An integer categorical label.
  ##
  ## Example:
  ##   one_hot_vec = c(0, 0, 0, 1, 0)
  ##   to_categorical(one_hot_vec)
  
  # Create categorical label.
  my_data_point = matrix(my_data_point, ncol = length(my_data_point))
  my_data_point = which(my_data_point == 1) - 1
  
}
