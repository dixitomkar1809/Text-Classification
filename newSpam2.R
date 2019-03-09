library(keras)
library(tm)
library(stringr)

# Read the input 
rawInputData <- read.csv("http://www.utdallas.edu/~ond170030/data/SMSSpamCollection.csv", header = TRUE, stringsAsFactors = FALSE)

# Taking just the first two columns of text and label
rawInputData <- subset(rawInputData, select = c('V1', 'V2'))
colnames(rawInputData) <- c("Class", "Text")

# Realized That the data contains N/A hence removed them
rawInputData <- t(na.omit(t(rawInputData)))

# Lets split the data into train and test and then we can go ahead with cleaning it
sample.size <- floor(0.70 * nrow(rawInputData))
set.seed(123)
trainIndex <- sample(seq_len(nrow(rawInputData)), size = sample.size)
trainData <- rawInputData[trainIndex, ]
testData <- rawInputData[-trainIndex, ]

# Now lets go ahead with cleaning the training data
corpusTrain <- VCorpus(VectorSource((trainData)))
corpusTrain <- tm_map(corpusTrain, removeNumbers)
corpusTrain <- tm_map(corpusTrain, removeWords, stopwords('english'))
corpusTrain <- tm_map(corpusTrain, removePunctuation)
corpusTrain <- tm_map(corpusTrain, stripWhitespace)
# corpusTrain <- tm_map(corpusTrain, stemDocument)

# Now lets go ahead with cleaning the test data
corpusTest <- VCorpus(VectorSource(testData))
corpusTest <- tm_map(corpusTest, removeNumbers)
corpusTest <- tm_map(corpusTest, removeWords, stopwords('english'))
corpusTest <- tm_map(corpusTest, removePunctuation)
corpusTest <- tm_map(corpusTest, stripWhitespace)
# corpusTest <- tm_map(corpusTest, stemDocument)

X1 <- c()
Y1 <- c()
for(i in 1:length(corpusTrain[[2]]$content)){
  if(str_length(corpusTrain[[2]]$content[i]) > 1){
    X1 <- c(X1, corpusTrain[[2]]$content[i])
    
    if(corpusTrain[[1]]$content[i] == "spam")
      Y1<-c(Y1,1)
    else
      Y1<-c(Y1,0)
  }
}

X2 <- c()
Y2 <- c()
for(i in 1:length(corpusTest[[2]]$content)){
  if(str_length(corpusTest[[2]]$content[i]) > 1){
    X2<-c(X2,corpusTest[[2]]$content[i])
    
    if(corpusTest[[1]]$content[i] == "spam")
      Y2<-c(Y2,1)
    else
      Y2<-c(Y2,0)
  }
}

vectorize_sequences <- function(sequences,
                                dimension = 10000) {
  # Create an all-zero matrix of shape
  #      (len(sequences), dimension)
  results <- matrix(0, nrow = length(sequences),
                    ncol = dimension)
  for (i in 1:length(sequences))
    
    # Sets specific indices of results[i] to 1s
    results[i, sequences[[i]]] <- 1
  
  results
} 
tokenizer <- text_tokenizer(num_words = 10000, filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n",
                            lower = TRUE, split = " ", char_level = TRUE, oov_token = NULL)%>%
  fit_text_tokenizer(X1)

trainSequences <- texts_to_sequences(tokenizer, X1)
testSequences <- texts_to_sequences(tokenizer, X2)

trainDataVectorized <- vectorize_sequences(trainSequences)
trainLabels <- as.numeric(Y1)

testDataVectorized <- vectorize_sequences(testSequences)
testLabels <- as.numeric(Y2)

model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu")%>%
  #layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
model %>% fit(trainDataVectorized, trainLabels, epochs = 4,
              batch_size = 512)
results <- model %>% evaluate(testDataVectorized, testLabels)

#---- eval=FALSE, message=FALSE------------------------------------------
results
