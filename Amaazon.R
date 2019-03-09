data <- readLines("http://utdallas.edu/~sxp175331/sentimentdataset/amazon_cells_labelled.txt")

fields<- strsplit(data, split = "\t")

library(tm)
for(i in 1:length(fields))
{
  fields[[i]][1]<-tolower(fields[[i]][1])
  fields[[i]][1]<-removeWords(fields[[i]][1], stopwords())#removes stopwords
  fields[[i]][1]<-gsub(pattern="\\W", replace=" ", fields[[i]][1]) #removes punctuation
  fields[[i]][1]<-gsub(pattern="\\d", replace=" ", fields[[i]][1]) #removes numbers
  fields[[i]][1]<-gsub(pattern="\\b[A-z]\\b{1}", replace=" ", fields[[i]][1]) #removes single letters  
  fields[[i]][1]<-stripWhitespace(fields[[i]][1]) #removes white spaces
  fields[[i]][1]<-trimws(fields[[i]][1])  #trims leading and trailing whitespaces
}


library(stringr)

k=1
word_list<-list()
label_list<-list()

for(i in 1:length(fields))
{
  temp_list<-str_split(fields[[i]][1], pattern="\\s+")
  j=1
  while(j<=length(temp_list[[1]]))
  {
    word_list[k]<-temp_list[[1]][j]
    label_list[k]<-fields[[i]][2]
    j=j+1
    k=k+1
    
  }
}

word_list
label_list
word_list<-unlist(word_list)
label_list<-unlist(label_list)
label_list<-as.numeric(label_list)

# Creates a tokenizer, configured to only take into account the 1,0000
# most common words, then builds the word index
tokenizer <- text_tokenizer(num_words = 10000) %>%
  fit_text_tokenizer(word_list)

# Turns strings into lists of integer indices
sequences <- texts_to_sequences(tokenizer, word_list)

newlist<-unlist(sequences)



require(caTools)
set.seed(101) 
sample = sample.split(newlist, SplitRatio = .75)

train_data = subset(newlist, sample == TRUE)
test_data  = subset(newlist, sample == FALSE)
train_labels = subset(label_list, sample == TRUE)
test_labels  = subset(label_list, sample == FALSE)



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


#---- eval=FALSE, message=FALSE------------------------------------------
# Our vectorized training data
x_train <- vectorize_sequences(train_data)
# Our vectorized test data
x_test <- vectorize_sequences(test_data)

# Vectorize Labels
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)


#---- eval=FALSE, message=FALSE------------------------------------------
library(keras)
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu",input_shape = c(10000)) %>%
  #layer_dense(units = 16, activation = "relu") 
  layer_dense(units = 1, activation = "sigmoid")

#---- eval=FALSE, message=FALSE------------------------------------------
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

model %>% fit(x_train, y_train, epochs = 12,
              batch_size = 512)
results <- model %>% evaluate(x_test, y_test)

#---- eval=FALSE, message=FALSE------------------------------------------
results




#---- eval=FALSE, message=FALSE------------------------------------------
val_indices <- 1:10000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

#---- eval=FALSE, message=FALSE------------------------------------------
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)