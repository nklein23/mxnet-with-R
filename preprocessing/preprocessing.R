################################################################################ 
################################ Preprocessing #################################
################################################################################ 
setwd("C:/Users/Niklas/mxnet-tutorials-in-R/data")
train.x = t(data.matrix(trainData))
train.y = numeric(length = dim(trainLabels)[1])

for(i in 1:9){
  train.y = train.y + i*trainLabels[, i]
}
