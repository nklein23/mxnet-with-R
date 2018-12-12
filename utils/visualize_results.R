visualize_results = function(my_training_results, custom_string, my_ylim) {
  
  ## A function to visualize the misclassification rate or the classification accuracy.
  ##  
  ## Args:
  ##   my_training_results (data.frame): A data frame containing either the training and 
  ##                                     validation accuracy, or the missclassification rate.
  ##                                     Must have two columns 'Train' and 'Test'.
  ##   custom_string (string): A custom string to be shown on the y-axis of the plot.
  ##   my_ylim (numeric): range of the y-axis, from 0 - my_ylim. 
  ##
  ## Returns:
  ##   A ggplot2 object.
  ##
  ## Example:
  ##   visualize_results(my_training_results, 'Accuracy')
  
  # Load the necessary packages.
  require('ggplot2')
  require('reshape2')
  
  # Create a data frame and melt it.
  training_results = as.data.frame(cbind(1:dim(my_training_results)[1], my_training_results))
  colnames(training_results)[1] = 'epoch'
  training_results = melt(training_results, id = 'epoch')
  
  # Convert to a factor.
  training_results$variable = factor(training_results$variable)
  
  # Generate the actual plot.
  ggplot(data = training_results, aes(x = epoch, y = value, colour = variable)) +
    geom_line(size = 1) +
    scale_y_continuous(name = custom_string, limits = c(0, my_ylim)) + 
    scale_x_continuous(labels = function (x) floor(x), name = 'epochs') + 
    labs(colour = '') + 
    theme_bw()
  
}

