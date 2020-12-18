library(arules)
library(arulesViz)
library(pacman)
library(tidyverse)
library(ggplot2)
library(corpus)
library("tm")
library("RTextTools")
library(SnowballC)
library(e1071)
library(dplyr)
library(wordcloud)
library(xgboost)
library(syuzhet)
library(ggthemes)

# read train
df <- read_csv("Desktop/train.csv")

nrow(df)
df_train <- slice(df, 1:(n()-25000))
nrow(df_train)

# read test
df2 <- read_csv("Desktop/train.csv")
df_test <- slice(df2, 1:(n()-25000))

nrow(df_test)
# read sample submission 
#sample_submission <- read_csv("../input//tweet-sentiment-extraction//sample_submission.csv")

# helper functions for text cleaning
removeHtmlTags <- function(x)
  (gsub("<.*?>", "", x))
removeHashTags <- function(x)
  gsub("#\\S+", " ", x)
removeTwitterHandles <- function(x)
  gsub("@\\S+", " ", x)
removeURL <- function(x)
  gsub("http:[[:alnum:]]*", " ", x)
removeApostrophe <- function(x)
  gsub("'", "", x)
removeNonLetters <- function(x)
  gsub("[^a-zA-Z\\s]", " ", x)
removeSingleChar <- function(x)
  gsub("\\s\\S\\s", " ", x)
# function to clean corpus
cleanCorpus <- function(reviews){
  # create the corpus
  corpus <- Corpus(VectorSource(reviews))
  # remove reviews
  rm(reviews)
  # remove twitter handles and hashtags
  corpus <- tm_map(corpus, content_transformer(removeHtmlTags))
  corpus <- tm_map(corpus,content_transformer(removeHashTags))
  corpus <- tm_map(corpus,content_transformer(removeTwitterHandles))
  # other cleaning transformations
  corpus <- tm_map(corpus, content_transformer(removeURL))
  corpus <- tm_map(corpus, content_transformer(removeApostrophe))
  corpus <- tm_map(corpus, content_transformer(removeNonLetters))
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, c(stopwords("english")))
  corpus <- tm_map(corpus, content_transformer(removeSingleChar))
  # Remove punctuations
  corpus <- tm_map(corpus, removePunctuation)
  # Eliminate extra white spaces
  corpus <- tm_map(corpus, stripWhitespace)    
  # stem document
  corpuse <- tm_map(corpus, stemDocument)
  return(corpuse)
}

# function get word frequency
wordFrequency <- function(corpus){
  dtm <- TermDocumentMatrix(corpus)
  rm(corpus)
  # convert to matrix
  m <- as.matrix(dtm)
  rm(dtm)
  # sort by word frequency
  v <- sort(rowSums(m),decreasing=TRUE)
  rm(m)
  # calculate word frequency
  word_frequencies <- data.frame(word = names(v),freq=v)
  return(word_frequencies)
}

# clean the tweets text
reviews <- df_train$text
#cor = Corpus(VectorSource(reviews)) 
corpus_train <- cleanCorpus(reviews)

inspect(corpus_train[1:9])
head(df_train)

# clean the tweets selected_text
reviews_selected_text <- df_train$selected_text
corpus_selected_text <- cleanCorpus(reviews_selected_text)


# split the data into by general sentiment
df_pos_train <- df_train[df_train$sentiment == "positive",]
df_neg_train <- df_train[df_train$sentiment == "negative",]
df_neu_train <- df_train[df_train$sentiment == "neutral",]


# make corpus
corpus_train_pos <- cleanCorpus(df_pos_train$text)
corpus_train_neg <- cleanCorpus(df_neg_train$text)
corpus_train_neu <- cleanCorpus(df_neu_train$text)

inspect(corpus_train_pos[1:5])

# calculate positive word frequency of Training Tweets
word_freq_train_pos <- wordFrequency(corpus_train_pos)
# print top 10 word frequencies
head(word_freq_train_pos, 10)

# calculate training tweets word frequency
word_frequencies_train <- wordFrequency(corpus_train)
# print top 10 word frequencies
head(word_frequencies_train, 10)

# plot wordcloud
# set random seed for reproducibility
set.seed(1234)
layout(matrix(c(1, 2), nrow=2), heights=c(0.6, 4))
par(mar=rep(0, 4))
plot.new()
text(x=0.5, y=0.1, "Overall Tweets Wordcloud of Training Data")
wordcloud(words = word_frequencies_train$word, freq = word_frequencies_train$freq, min.freq = 10,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))


# Document term matrix
dtm= DocumentTermMatrix(corpus_train)
train_mat= as.matrix(dtm)
head(train_mat[1:5, 1:8])
anyNA(train_mat)


# Implement trainControl method for computational overheads
# To fix the sample 
set.seed(16102016)  

# From the training dataset select 70% of the tweets for training and 30% to validate the results
samp_id = sample(1:nrow(df_train),              
                 round(nrow(df_train)*.70),     
                 replace = F)               # sampling without replacement.

train = df_train[samp_id, ]                      # 70% of training data set, examine struc of samp_id obj
test = df_train[-samp_id,]                      # remaining 30% of training data set


#Process the text data and create DTM (Document Term Matrix)

train.data = rbind(train,test)              # join the data sets
train.data$text = tolower(train.data$text)  # Convert to lower case
text = train.data$text                      
text = removePunctuation(text)              # remove punctuation marks
text = removeNumbers(text)                  # remove numbers
text = stripWhitespace(text)                # remove blank space

# custom code for removing custom stopwords from your textfile
#stpw1 = readLines(".../stopwords_Twitter.txt")# stopwords list from git
#stpw2 = tm::stopwords('english')               # tm package stop word list; tokenizer package has the same name function

#comn  = unique(c(stpw1, stpw2))                 # Union of two list #'solid state chemistry','microeconomics','linguistic'
#stopwords = unique(gsub("'"," ",comn))  # final stop word lsit after removing punctuation
#text  =  removeWords(text,tm::stopwords('english')) 

# Create text corpus
cor = Corpus(VectorSource(text))         

# Craete DTM
dtm = DocumentTermMatrix(cor,               
                         control = list(weighting =             
                                          function(x)
                                            weightTfIdf(x, normalize = F))) # IDF weighing

# Coded labels- Target varaible Sentiment
training_codes = train.data$sentiment     

####  # creates a 'container' obj for training, classifying, and analyzing docs
container <- create_container(dtm,              
                              t(training_codes), # labels or the Y variable / outcome we want to train on
                              trainSize = 1:nrow(train), 
                              testSize = (nrow(train)+1):nrow(train.data), 
                              virgin = FALSE)

# Check available algorithms
#print_algorithms() 

# train_models; makes a model object using the specified algorithms.
models <- train_models(container,              
                       algorithms=c("BAGGING"))   #"SVM","GLMNET","SLDA","TREE","BAGGING","BOOSTING","RF"




# training data results
results <- classify_models(container, models)


# building a confusion matrix to see accuracy of prediction results for training data
out_train = data.frame(model_sentiment = results$SVM_LABEL,    # rounded probability == model's prediction of Y
                 model_prob = results$SVM_PROB,
                 actual_sentiment = train.data$sentiment[(nrow(train)+1):nrow(train.data)])  # actual value of Y

# dim(out); head(out); 
# summary(out)           # how many 0s and 1s were there anyway?

# display the confusion matrix for taining data
(z = as.matrix(table(out_train[,1], out_train[,3])))   


# prediction accuracy in % terms for training data
(pct = round(((z[1,1] + z[2,2] + z[3,3])/sum(z))*100, 2))  



set.seed(16102016)
text = df_test$text
text = removePunctuation(text)
text = removeNumbers(text)
text = stripWhitespace(text)
cor = Corpus(VectorSource(text))
dtm.test = DocumentTermMatrix(cor, control = list(weighting = 
                                                    function(x)
                                                      weightTfIdf(x, normalize = F)))

row.names(dtm.test) = (nrow(dtm)+1):(nrow(dtm)+nrow(dtm.test))     # row naming for doc ID
dtm.f = c(dtm, dtm.test)    # concatenating the dtms
training_codes.f = c(training_codes, 
                     rep(NA, length(df_test))) 

container.f = create_container(dtm.f,      # build a new container; all same as before
                               t(training_codes.f), trainSize=1:nrow(dtm), 
                               testSize = (nrow(dtm)+1):(nrow(dtm)+nrow(dtm.test)), virgin = T)

model.f = train_models(container.f, algorithms = c("TREE")) 

predicted <- classify_models(container.f, model.f)     # ?classify_models makes predictions from a train_models() object.

# predicted output
out_test = data.frame(model_sentiment = predicted$SVM_LABEL,    
                 model_prob = predicted$SVM_PROB,
                 text = df_test)


# display the confusion matrix for test data
(z = as.matrix(table(out_test[,1], out_test[,6])))   

# Accuracy for test data
# prediction accuracy in % terms
(pct = round(((z[1,1] + z[2,2] + z[3,3])/sum(z))*100, 2))  

# Sample predicted output
head(out,10)

# Visual representation of Conclusion for overall sentiments of tweets
emotion_selected <- get_nrc_sentiment(df_train$selected_text)
emotion_df_selected <- as.data.frame(colSums(emotion_selected))
emotion_df_selected <- rownames_to_column(emotion_df_selected) 
colnames(emotion_df_selected) <- c("emotion", "count")
ggplot(emotion_df_selected, aes(x =emotion, y = count, fill = emotion)) + geom_bar(stat = "identity") + 
   theme(legend.position="none", panel.grid.major = element_blank()) + labs( x = "Emotion", y = "Total Count") +
  ggtitle("Sentiment of Overall Tweets") + theme(plot.title = element_text(hjust=0.5))+coord_flip()
  
  