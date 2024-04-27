setwd("C:/Users/Dhruv/Desktop/Projects Q3 2023/(3) Sentiment analysis/Threads app reviews")
library(tidyverse)
library(tidytext)
library(textdata)
library(tm)
library(wordcloud)
library(tm)
library(slam)
library(caret)
library(smotefamily)
library(ROSE)
library(quanteda)
data <- read.csv("threads_reviews.csv")

unique(data$source)
unique(data$rating)

view(filter(data, data$rating == "3"))
2585/32910

view(filter(data, data$source == "App Store"))
2640/32910

summary(data)


#distribution of ratings

ggplot(data, aes(x = rating)) + 
  geom_bar(aes(fill = source)) +
  facet_wrap(~ source)


##################################################################################################################

#(i) Lexicon-Based Sentiment Analysis (Absolute Method)
reviews_data_tokens <- data %>%
  unnest_tokens(word, review_description)

sentiments <- reviews_data_tokens %>%
  inner_join(get_sentiments("afinn")) %>%
  group_by(source, rating) %>%
  summarise(sentiment_score = sum(value)) %>%
  arrange(desc(sentiment_score))


view(sentiments)

#####################################################################################################################
# Sentiments not rating wise:
sentiments_no_rating <- reviews_data_tokens %>%
  inner_join(get_sentiments("afinn")) %>%
  group_by(source) %>%
  summarise(sentiment_score = sum(value)) %>%
  arrange(desc(sentiment_score))

view(sentiments_no_rating)


#VIZ:
#1
ggplot(sentiments, aes(x=rating, y=sentiment_score, fill=source)) +
  geom_bar(stat='identity') +
  facet_wrap(~ source) +
  labs(title="Absolute Sentiment Scores by Rating and Source")

#2
ggplot(sentiments_no_rating, aes(x=source, y=sentiment_score, fill=source)) +
  geom_bar(stat='identity') +
  labs(title="Absolute Sentiment Scores by Source")



#(ii) Lexicon-Based Sentiment Analysis (Normalised by count method)

# Compute the number of reviews per combination of source and rating
review_count <- reviews_data_tokens %>%
  group_by(source, rating) %>%
  summarise(n = n())

# Compute average sentiment scores
sentiments_normalized <- sentiments %>%
  inner_join(review_count, by = c("source", "rating")) %>%
  mutate(average_sentiment = sentiment_score / n)

view(sentiments_normalized)

####
#average sentiment scores without rating
review_count_no_rating <- reviews_data_tokens %>%
  group_by(source) %>%
  summarise(n = n())


sentiments_normalized_no_rating <- sentiments_no_rating %>%
  inner_join(review_count_no_rating, by = ("source")) %>%
  mutate(average_sentiment = sentiment_score / n)

view(sentiments_normalized_no_rating)
###################################################################################################################


#VIZ:
#1
ggplot(sentiments_normalized, aes(x=rating, y=average_sentiment, fill=source)) +
  geom_bar(stat='identity') +
  facet_wrap(~ source) +
  labs(title="Average Sentiment Scores by Rating and Source")

#2
ggplot(sentiments_normalized_no_rating, aes(x=source, y=average_sentiment, fill=source)) +
  geom_bar(stat='identity') +
  labs(title="Average Sentiment Scores by Source")


################################################################################################################
#(iii) Lexicon-Based Sentiment Analysis (Proportional Sentiment Score method)
# Compute the number of tokens per combination of source and rating
token_count <- reviews_data_tokens %>%
  group_by(source, rating) %>%
  summarise(token_n = n())

# Compute sentiment scores per token
sentiments_token_normalized <- sentiments %>%
  inner_join(token_count, by = c("source", "rating")) %>%
  mutate(sentiment_per_token = sentiment_score / token_n)


#Viz:
# Plotting the sentiment score per token
ggplot(sentiments_token_normalized, aes(x=rating, y=sentiment_per_token, fill=source)) +
  geom_bar(stat='identity') +
  facet_wrap(~ source) +
  labs(title="Sentiment Scores per Token by Rating and Source")

#ITS IDENTICAL SO I'LL JUST USE METHOD (II) NORMALISED BY COUNT

##############################################################################################################################
#KEYWORD ANALYSIS:

# Create a corpus
reviews_corpus <- Corpus(VectorSource(data$review_description))

# Remove additional stop words
additional_stopwords <- c("app", "threads", "just", "account", "see", "cant", 
                          "people", "follow", "want", "post", "one", 
                          "copy", "need", "thread", "even", "also", "really", "now", "option", "delete", 
                          "much", "’s", "first", "far", "way", "get", "make", "accounts", 
                          "meta", "still", "application", "fix", "\U0001f44d", "elon", "without", "social", 
                          "think", "apps", "don’t", "know", "following", "using", "something", 
                          "insta", "lot", "mark", "download", "well", 
                          "open", "musk", "back", "another", 
                          "already", "nothing", "thing", "anything", 
                          "every","keeps", "keep", "things", "’m", 
                          "let", "start",  "thanks", "log", "going",
                          "everything", "looks", "ever", "zuck", "since", 
                          "right", "theres", "soon", "find", "seems", "can’t", "aap", 
                          "getting", "day", "full", "got", "trying", "thank", "tried", 
                          "thats", "say", "never",  
                          "yet", "actually", "everyone", "sign", "made", "ive", 
                          "used", "look", "days", "video", "stars", "feels", 
                          "didnt", "install", "feel", "switch", "sure", 
                          "though", "seeing", "times", "overall", "lets", 
                          "always", "looking", "tab", "showing", "wont", 
                          "reason", "\U0001f602", "makes", "name", 
                          "point", "literally", "care", "fine", 
                          "making", "come", "shows", "ill", "stuff", "maybe", "speech", "big", "team",
                          "etc", "however", "someone", "idea", "definitely", "multiple", 
                          "world", "lol", "whenever", "take", "reply", "hard",  
                          "\U0001f60d", "stop", "others", "little", "\U0001f60a", "dont", "will", "time", "try", "work", "hai",
                          "twitter","instagram","facebook","zuckerberg","like")

# Merge with the standard stopwords
all_stopwords <- c(stopwords("en"), additional_stopwords)


words_in_top_300_that_will_be_in_list <- c("good", "nice", "better", "use", "can", "great", 
                                           "please", "new", "best", "love", "feed", "add", "features", "posts", "bad", 
                                           "amazing", "experience", "needs", "data", "media", "hope", "many", "cool", 
                                           "feature", "working", "user", "content", "doesnt", "give", "page", "review", "easy", 
                                           "profile", "able", "users", "phone", "screen", "super", "worst", "problem", 
                                           "awesome", "search", "login", "bugs", "platform", "crashing", "bug", "crashes", "version", 
                                           "interface", "scroll", "upload", "trending", "followers", "useless", "random", "dark", 
                                           "wow", "excellent", "update", "mode", "button", "hashtags", "create", "pretty", 
                                           "change", "glitch", "boring", "friends", "star", "glitches", "text", "issue", "videos", 
                                           "timeline", "show", "photos", "home", "deleting", "downloaded", "fun", "properly", "save", 
                                           "wish", "alternative", "different", "message", "instead", "simple", "missing", "interesting", 
                                           "photo", "waste", "edit", "privacy", "deleted", "future", "glitching", "works", "perfect", "help",
                                           "share", "paste", "android", "allow", "❤️", "algorithm", "smooth", "updates", "scrolling", "support", "personal", 
                                           "less", "installed", "wrong", "view", "error", "annoying", "pictures", "picture", "free", "information", "fixed", 
                                           "trash", "unable", "badge", "cheap", "comment", "read", "issues", "posting", "access", "clean", "interested", 
                                           "topics", "top", "useful")




# Transforming the data by converting to lowercase, removing punctuation, numbers, whitespaces, and stop words
reviews_corpus_clean <- tm_map(reviews_corpus, content_transformer(tolower))
reviews_corpus_clean <- tm_map(reviews_corpus_clean, removePunctuation)
reviews_corpus_clean <- tm_map(reviews_corpus_clean, removeNumbers)
reviews_corpus_clean <- tm_map(reviews_corpus_clean, removeWords, all_stopwords)
reviews_corpus_clean <- tm_map(reviews_corpus_clean, stripWhitespace)


word_freq <- TermDocumentMatrix(reviews_corpus_clean)
word_freq <- as.data.frame(as.matrix(word_freq))
word_freq_sum <- rowSums(word_freq, na.rm = TRUE)
word_freq_df <- data.frame(term = names(word_freq_sum), freq = word_freq_sum)

# Viewing the df:
word_freq_df_ordered <- word_freq_df[rev(order(word_freq_df$freq)),]
view(word_freq_df_ordered)
view(head(word_freq_df_ordered, 15))


# Word cloud
wordcloud(words = word_freq_df$term, freq = word_freq_df$freq, min.freq = 50,
          max.words=80, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"), scale=c(3, 0.5))





# Word clouds by positive and negative reviews
# Since the 3 star reviews are only 7-8% of the total dataset, we can ignore these:

# Defining Positive and Negative Reviews
positive_reviews <- subset(data, rating %in% c(4, 5))
negative_reviews <- subset(data, rating %in% c(1, 2))



# Create a corpus
positive_corpus <- Corpus(VectorSource(positive_reviews$review_description))

# Transform the data
positive_corpus_clean <- tm_map(positive_corpus, content_transformer(tolower))
positive_corpus_clean <- tm_map(positive_corpus_clean, removePunctuation)
positive_corpus_clean <- tm_map(positive_corpus_clean, removeNumbers)
positive_corpus_clean <- tm_map(positive_corpus_clean, removeWords, all_stopwords)
positive_corpus_clean <- tm_map(positive_corpus_clean, stripWhitespace)



# Create a corpus
negative_corpus <- Corpus(VectorSource(negative_reviews$review_description))

# Transform the data
negative_corpus_clean <- tm_map(negative_corpus, content_transformer(tolower))
negative_corpus_clean <- tm_map(negative_corpus_clean, removePunctuation)
negative_corpus_clean <- tm_map(negative_corpus_clean, removeNumbers)
negative_corpus_clean <- tm_map(negative_corpus_clean, removeWords, all_stopwords)
negative_corpus_clean <- tm_map(negative_corpus_clean, stripWhitespace)


# Positive reviews word cloud
positive_freq <- TermDocumentMatrix(positive_corpus_clean)
positive_freq <- as.data.frame(as.matrix(positive_freq))
positive_freq_sum <- rowSums(positive_freq, na.rm = TRUE)
positive_freq_df <- data.frame(term = names(positive_freq_sum), freq = positive_freq_sum)

# Viewing the table:
positive_freq_df_ordered <- positive_freq_df[rev(order(positive_freq_df$freq)),]
view(head(positive_freq_df_ordered, 15))

# Plotting the word cloud
wordcloud(words = positive_freq_df$term, freq = positive_freq_df$freq, min.freq = 50,
          max.words=85, random.order=FALSE, rot.per=0.35, colors=brewer.pal(8, "Dark2"), scale=c(3, 0.5))



# Negative reviews word cloud
negative_freq <- TermDocumentMatrix(negative_corpus_clean)
negative_freq <- as.data.frame(as.matrix(negative_freq))
negative_freq_sum <- rowSums(negative_freq, na.rm = TRUE)
negative_freq_df <- data.frame(term = names(negative_freq_sum), freq = negative_freq_sum)

# Viewing the table:
negative_freq_df_ordered <- negative_freq_df[rev(order(negative_freq_df$freq)),]
view(head(negative_freq_df_ordered, 15))


# Plotting the word cloud
# Adjust plot margins
par(mar=c(4, 4, 4, 4))

wordcloud(words = negative_freq_df$term, freq = negative_freq_df$freq, min.freq = 50,
          max.words=70, random.order=FALSE, rot.per=0.35, colors=brewer.pal(8, "Dark2"), scale=c(2.5, 0.5))




##########################################################################################################################
# Remove additional stop words
additional_stopwords <- c("app", "threads", "just", "account", "see", "cant", 
                          "people", "follow", "want", "post", "one", 
                          "copy", "need", "thread", "even", "also", "really", "now", "option", "delete", 
                          "much", "’s", "first", "far", "way", "get", "make", "accounts", 
                          "meta", "still", "application", "fix", "\U0001f44d", "elon", "without", "social", 
                          "think", "apps", "don’t", "know", "following", "using", "something", 
                          "insta", "lot", "mark", "download", "well", 
                          "open", "musk", "back", "another", 
                          "already", "nothing", "thing", "anything", 
                          "every","keeps", "keep", "things", "’m", 
                          "let", "start",  "thanks", "log", "going",
                          "everything", "looks", "ever", "zuck", "since", 
                          "right", "theres", "soon", "find", "seems", "can’t", "aap", 
                          "getting", "day", "full", "got", "trying", "thank", "tried", 
                          "thats", "say", "never",  
                          "yet", "actually", "everyone", "sign", "made", "ive", 
                          "used", "look", "days", "video", "stars", "feels", 
                          "didnt", "install", "feel", "switch", "sure", 
                          "though", "seeing", "times", "overall", "lets", 
                          "always", "looking", "tab", "showing", "wont", 
                          "reason", "\U0001f602", "makes", "name", 
                          "point", "literally", "care", "fine", 
                          "making", "come", "shows", "ill", "stuff", "maybe", "speech", "big", "team",
                          "etc", "however", "someone", "idea", "definitely", "multiple", 
                          "world", "lol", "whenever", "take", "reply", "hard",  
                          "\U0001f60d", "stop", "others", "little", "\U0001f60a", "dont", "will", "time", "try", "work", "hai",
                          "twitter","instagram","facebook","zuckerberg","like")

# Merge the additional stop words with the default English stop words
all_stopwords <- c(stopwords("en"), additional_stopwords)

is.character(all_stopwords) 
#[1] TRUE


################################################################################################################################################

# Machine learning model:
# (i) creating TF-IDF matrix
data$sentiment <- case_when(
  data$rating >= 4 ~ "positive",
  data$rating == 3 ~ "neutral",
  data$rating <= 2 ~ "negative"
)



# Create a text corpus
corpus = Corpus(VectorSource(data$review_description))

# Convert to lower-case
corpus = tm_map(corpus, content_transformer(tolower))

# Remove numbers
corpus = tm_map(corpus, removeNumbers)

# Remove special characters
corpus = tm_map(corpus, removePunctuation)

# Remove stop words, using the combined list of stop words
corpus = tm_map(corpus, removeWords, all_stopwords)

# Strip white space
corpus = tm_map(corpus, stripWhitespace)

# Create a Document-Term Matrix (DTM)
dtm = DocumentTermMatrix(corpus)

# Remove sparse terms
dtm <- removeSparseTerms(dtm, 0.995)  

# Compute the Term Frequency-Inverse Document Frequency (TF-IDF)
tfidf = weightTfIdf(dtm)

#convert it to a regular matrix afterward 
tfidf_matrix = as.matrix(tfidf)

# Convert to data frame
tfidf_df = as.data.frame(as.matrix(tfidf))

# Combine with other variables, like 'source' and 'sentiment'
final_data = data.frame(source=data$source, sentiment=data$sentiment, tfidf_df)

###################################################################################################################
#1st attempt:

# Splitting data into training and testing:
set.seed(123)
trainIndex <- createDataPartition(final_data$sentiment, p=0.8, list=FALSE)
trainData <- final_data[trainIndex,]
testData <- final_data[-trainIndex,]


# Model training:
set.seed(123)
model <- train(sentiment ~ ., data=trainData, method="naive_bayes")

# Evaluating model:
# Make predictions
predictions <- predict(model, newdata=testData)

predictions <- as.factor(predictions)
testData$sentiment <- as.factor(testData$sentiment)

# Evaluate the model
confusionMatrix(predictions, testData$sentiment)

# Rrsults are not great, model performs very badly on neutral and negative classes.This is because positive classes make up the bulk of the set
# Attempting this again by oversampling for neutral and negative:
###############################################################################################################################################################################################

# Attempt 2
# Looking at the actual size for each class:
# Calculate the count for each class
class_counts <- final_data %>%
  group_by(sentiment) %>%
  summarise(Count = n())
class_counts



# Current sizes
size_negative <- 11522
size_neutral <- 2585
size_positive <- 18803


average_size <- 10970

# Calculate new target size for the neutral class
# This example targets halfway between its current size and the average:
target_size_neutral <- round((size_neutral + average_size) / 2)

# No change to negative class size
target_size_negative <- size_negative

# Separate the dataset into parts based on sentiment
data_negative <- filter(final_data, sentiment == "negative")
data_neutral <- filter(final_data, sentiment == "neutral")
data_positive <- filter(final_data, sentiment == "positive")

# Oversample neutral class to the new target size
data_neutral_oversampled <- sample_n(data_neutral, target_size_neutral, replace = TRUE)
data_negative_oversampled <- sample_n(data_negative, target_size_negative, replace = TRUE) 

# Combine back with the positive
final_data_oversampled <- bind_rows(data_negative_oversampled, data_neutral_oversampled, data_positive)

# Shuffle the rows to mix the data well
set.seed(123) 
final_data_oversampled <- final_data_oversampled[sample(nrow(final_data_oversampled)),]


# Model training on 'final_data_oversampled'
set.seed(123)
trainIndex_v2 <- createDataPartition(final_data_oversampled$sentiment, p=0.8, list=FALSE)
trainData_v2 <- final_data_oversampled[trainIndex_v2,]
testData_v2 <- final_data_oversampled[-trainIndex_v2,]


# Model training:
set.seed(123)
model_v2 <- train(sentiment ~ ., data=trainData_v2,method="naive_bayes")

# Evaluating model:
# Make predictions
predictions_v2 <- predict(model_v2, newdata=testData_v2)

predictions_v2 <- as.factor(predictions_v2)
testData_v2$sentiment <- as.factor(testData_v2$sentiment)

# Evaluate the model
confusionMatrix(predictions_v2, testData_v2$sentiment)


# The overall accuracy is actually less now, perhaps manually dealing with the negative and neutral classes was not the best approach
# I will now try the oversampling techniques of SMOTE for more nuanced class balancing
############################################################################################################################################################################################################################################



# Attempt 3: simply classifying 'neutral' as 'negative':

# Creating TF-IDF matrix
data$sentiment <- case_when(
  data$rating >= 4 ~ "positive",
  data$rating <= 3 ~ "negative"
)



# Create a text corpus
corpus = Corpus(VectorSource(data$review_description))

# Convert to lower-case
corpus = tm_map(corpus, content_transformer(tolower))

# Remove numbers
corpus = tm_map(corpus, removeNumbers)

# Remove special characters
corpus = tm_map(corpus, removePunctuation)

# Remove stop words, using the combined list of stop words
corpus = tm_map(corpus, removeWords, all_stopwords)

# Strip white space
corpus = tm_map(corpus, stripWhitespace)

# Create a Document-Term Matrix (DTM)
dtm = DocumentTermMatrix(corpus)

# Remove sparse terms
dtm <- removeSparseTerms(dtm, 0.995)  

# Compute the Term Frequency-Inverse Document Frequency (TF-IDF)
tfidf = weightTfIdf(dtm)

# regular matrix  
tfidf_matrix = as.matrix(tfidf)


# Convert to data frame
tfidf_df = as.data.frame(as.matrix(tfidf))

# Combine with other variables, like 'source' and 'sentiment'
final_data = data.frame(source=data$source, sentiment=data$sentiment, tfidf_df)

set.seed(123)
trainIndex <- createDataPartition(final_data$sentiment, p=0.8, list=FALSE)
trainData <- final_data[trainIndex,]
testData <- final_data[-trainIndex,]


# Model training:
set.seed(446)
model <- train(sentiment ~ ., data=trainData, method="naive_bayes")

# Evaluating model:
# Make predictions
predictions <- predict(model, newdata=testData)

predictions <- as.factor(predictions)
testData$sentiment <- as.factor(testData$sentiment)

# Evaluate the model
confusionMatrix(predictions, testData$sentiment)
################################################################################################################################################################
#Attempt 4 (not able to run): bigrams for 'negation'
# Machine learning model:
# Adjusting sentiment to binary classification
# Adjust the sentiment classification to binary


data$sentiment <- ifelse(data$rating >= 4, "positive", "negative")

# Create a tokens object, now including both unigrams and bigrams
tokens <- tokens(data$review_description, what = "word", remove_punct = TRUE, remove_numbers = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(pattern = all_stopwords) %>%
  tokens_ngrams(n = 2)  # Generate bigrams

# Create a document-feature matrix (DFM) from tokens and remove infrequent terms
dfm <- dfm(tokens) %>%
  dfm_trim(min_termfreq = 0.98, termfreq_type = "quantile", max_docfreq = 0.5, docfreq_type = "prop")

# Compute TF-IDF weights
tfidf <- dfm_tfidf(dfm)

# Convert the DFM to a data frame for modeling; this is memory efficient and avoids issues with large matrices
tfidf_df <- convert(tfidf, to = "data.frame")

# Add document identifiers for merging
tfidf_df$doc_id <- rownames(tfidf_df)

# Merge the TF-IDF features with the original data
final_data <- merge(data, tfidf_df, by.x = "row.names", by.y = "doc_id", all.x = TRUE)

# Ensure that 'sentiment' is a factor for the modeling
final_data$sentiment <- as.factor(final_data$sentiment)

# Splitting data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(final_data$sentiment, p = 0.8, list = FALSE)
trainData <- final_data[trainIndex, ]
testData <- final_data[-trainIndex, ]


# Model training using Naive Bayes
set.seed(123)
model <- train(sentiment ~ ., data = trainData, method = "naive_bayes", trControl = trainControl(method = "cv", number = 10))

# Making predictions on the test set
predictions <- predict(model, newdata = testData)

# Evaluating the model
confusionMatrix(predictions, testData$sentiment)

################################################################################################################################################################################

#Attempt 5: handling negation in the least computationally intensive way:
# Define the custom negation handling function
handle_negation <- function(text) {
  text_modified <- gsub("not (\\w+)", "not_\\1", text)
  return(text_modified)
}

# Apply negation handling
data$review_description <- sapply(data$review_description, handle_negation)

# Adjust sentiment based on rating
data$sentiment <- case_when(
  data$rating >= 4 ~ "positive",
  data$rating <= 3 ~ "negative"
)

# Load text data into a corpus
corpus <- Corpus(VectorSource(data$review_description))

# Preprocess the corpus
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, content_transformer(removePunctuation))
corpus <- tm_map(corpus, removeWords, all_stopwords)
corpus <- tm_map(corpus, stripWhitespace)

# Create a Document-Term Matrix (DTM)
dtm <- DocumentTermMatrix(corpus)

# Apply term frequency filtering to reduce the feature space
dtm <- removeSparseTerms(dtm, 0.995)  

# Compute the Term Frequency-Inverse Document Frequency (TF-IDF)
tfidf <- weightTfIdf(dtm)

# Convert TF-IDF to a data frame
tfidf_df <- as.data.frame(as.matrix(tfidf), stringsAsFactors = FALSE)

# Prepare final_data by combining 'source', 'sentiment' and TF-IDF features
final_data <- cbind(data[, c("source", "sentiment")], tfidf_df)

# Ensure 'sentiment' is correctly formatted as a factor for modeling
final_data$sentiment <- as.factor(final_data$sentiment)

# Splitting data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(final_data$sentiment, p = 0.8, list = FALSE)
trainData <- final_data[trainIndex, ]
testData <- final_data[-trainIndex, ]

# Model training using Naive Bayes
set.seed(123)
model <- train(sentiment ~ ., data = trainData, method = "naive_bayes", trControl = trainControl(method = "cv", number = 10))

# Making predictions on the test set
predictions <- predict(model, newdata = testData)

# Evaluating the model
confusionMatrix(predictions, testData$sentiment)









######################################################################################################################

#Results of 1st attempt:
#Confusion Matrix and Statistics

#Reference
#Prediction negative neutral positive
#negative      519      66      144
#neutral       344     178      316
#positive     1441     273     3300

#Overall Statistics

#Accuracy : 0.6074          
#95% CI : (0.5954, 0.6192)
#No Information Rate : 0.5713          
#P-Value [Acc > NIR] : 1.673e-09       

#Kappa : 0.2389          

#Mcnemar's Test P-Value : < 2.2e-16       

#Statistics by Class:

#                     Class: negative Class: neutral Class: positive
#Sensitivity                  0.22526        0.34429          0.8777
#Specificity                  0.95090        0.89116          0.3924
#Pos Pred Value               0.71193        0.21241          0.6582
#Neg Pred Value               0.69498        0.94097          0.7064
#Prevalence                   0.35010        0.07856          0.5713
#Detection Rate               0.07886        0.02705          0.5014
#Detection Prevalence         0.11077        0.12734          0.7619
#Balanced Accuracy            0.58808        0.61773          0.6350


#Results of 2nd attempt (oversampling):
#Confusion Matrix and Statistics

#Reference
#Prediction negative neutral positive
#negative      495     157      215
#neutral       363     453      302
#positive     1446     745     3243

#Overall Statistics

#Accuracy : 0.5649          
#95% CI : (0.5535, 0.5762)
#No Information Rate : 0.5068          
#P-Value [Acc > NIR] : < 2.2e-16       

#Kappa : 0.2299          

#Mcnemar's Test P-Value : < 2.2e-16       

#Statistics by Class:

#                    Class: negative Class: neutral Class: positive
#Sensitivity                  0.21484        0.33432          0.8625
#Specificity                  0.92727        0.89034          0.4012
#Pos Pred Value               0.57093        0.40519          0.5968
#Neg Pred Value               0.72390        0.85685          0.7395
#Prevalence                   0.31055        0.18264          0.5068
#Detection Rate               0.06672        0.06106          0.4371
#Detection Prevalence         0.11686        0.15069          0.7324
#Balanced Accuracy            0.57106        0.61233          0.6319

#Results of 3rd attempt (removing neutral):

#Confusion Matrix and Statistics

#Reference
#Prediction negative positive
#negative     1007      375
#positive     1814     3385

#Accuracy : 0.6674          
#95% CI : (0.6558, 0.6788)
#No Information Rate : 0.5713          
#P-Value [Acc > NIR] : < 2.2e-16       

#Kappa : 0.2747          

#Mcnemar's Test P-Value : < 2.2e-16       

#            Sensitivity : 0.3570          
#            Specificity : 0.9003          
#         Pos Pred Value : 0.7287          
#         Neg Pred Value : 0.6511          
#             Prevalence : 0.4287          
#         Detection Rate : 0.1530          
#   Detection Prevalence : 0.2100          
#      Balanced Accuracy : 0.6286          

#      'Positive' Class : negative   





#results of 4th attempt:
#Confusion Matrix and Statistics

#Reference
#Prediction negative positive
#negative      999      360
#positive     1822     3400

#Accuracy : 0.6684          
#95% CI : (0.6569, 0.6798)
#No Information Rate : 0.5713          
#P-Value [Acc > NIR] : < 2.2e-16       

#Kappa : 0.2763          

#Mcnemar's Test P-Value : < 2.2e-16       

#           Sensitivity : 0.3541          
#          Specificity : 0.9043          
#      Pos Pred Value : 0.7351          
#     Neg Pred Value : 0.6511          
#        Prevalence : 0.4287          
#   Detection Rate : 0.1518          
#Detection Prevalence : 0.2065          
#  Balanced Accuracy : 0.6292          

#  'Positive' Class : negative   



