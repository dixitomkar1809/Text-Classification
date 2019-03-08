library(keras)
library(tm)
library(tokenizers)
library(stopwords)
library(text2vec)
library(SnowballC)
library(plotly)

# Reading Data
# inputData <- read.table(file ="./SMSSpamCollection", sep="\t", fill = FALSE, col.names = c("Class", "Text"))
inputData <- read.csv(file ='SMSSpamCollection.csv', header = FALSE, sep = ',')

# Taking just the first two columns of text and label
inputData <- subset(inputData, select = c('V1', 'V2'))
colnames(inputData) <- c("Class", "Text")

# Converting Legit message to 1 and spam to 0
inputData$Class <- ifelse(inputData$Class=='ham', 1, 0)

# Converting to Corpus
docs <- Corpus(VectorSource(inputData$Text))
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removeWords, stopwords('english'))
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, stripWhitespace)
docs <- tm_map(docs, stemDocument)

inputData$Text <- docs$content

hist(inputData$Class)
