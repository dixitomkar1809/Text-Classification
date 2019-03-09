library(keras)
library(tm)
library(tokenizers)
library(stopwords)
library(text2vec)
library(SnowballC)
library(plotly)

# Reading Data
# inputData <- read.table(file ="./SMSSpamCollection", sep="\t", fill = FALSE, col.names = c("Class", "Text"))
inputData <- read.csv(file ='./SMSSpamCollection.csv', header = FALSE, sep = ',')

# Taking just the first two columns of text and label
inputData <- subset(inputData, select = c('V1', 'V2'))
colnames(inputData) <- c("Class", "Text")

# Converting Legit message to 1 and spam to 0
inputData$Class <- ifelse(inputData$Class=='ham', 1, 0)

# Distribution of Legit and Spam
prop.table(table(inputData$Class))

# Converting to Corpus
corpus <- Corpus(VectorSource(inputData$Text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords('english'))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, stemDocument)

# Creating Document Term Matrix
dtm = DocumentTermMatrix(corpus)

# Creating test train dtm
dtm_train <- dtm[1:round(nrow(dtm)*0.80, 0), ]
dtm_test <- dtm[(round(nrow(dtm)*0.80, 0)+1):nrow(dtm), ]

# Taking Train test labels
train_labels <- inputData[1:round(nrow(inputData)*0.80, 0), ]$Class
test_labels <- inputData[(round(nrow(inputData)*0.80, 0)+1):nrow(dtm), ]$Class

# Checking the proportion of legit and spam messages
prop.table(table(train_labels))
prop.table(table(test_labels))

# calculating Minimum Frequency
threshold <- 0.1
min_freq <- round(dtm$nrow*(threshold/100), 0)

# Taking Most Frequency Words
most_frequent_words <- findFreqTerms(dtm, lowfreq = min_freq)

dtm_train_most_frequent <- dtm_train[, most_frequent_words]
dtm_test_most_frequent <- dtm_test[, most_frequent_words]

dim(dtm_train_most_frequent)

# Convert to Numeric values
# if the matrix says 0 we want to say no else yes regardess of the count
toNum <- function(x){
  x <-ifelse(test=x > 0, yes="Yes", no="No")
}

dtm_train_most_frequent <-apply(dtm_train_most_frequent, MARGIN = 2, FUN = toNum)
dtm_test_most_frequent <- apply(dtm_test_most_frequent, MARGIN = 2, FUN = toNum)

model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu",
              input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)