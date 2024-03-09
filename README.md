# "Threads" Sentiment Analysis

## Project Overview
This analysis dives into over 32,000 reviews of the "Threads" app from both the App Store and Google Play Store, aiming to uncover user sentiments and insights through sentiment analysis. (Add more about ML).

## Analysis Approach:

(I) Exploratory analysis - platform-wise distribution of ratings across these reviews.

(ii) Sentiment scores - Utilized predefined lists from R packages to assign sentiment scores to reviews based on positive or negative word associations.

(iii) Keyword Analysis

(iv) Machine Learning Models - Leverage review data to build machine learning models capable of sentiment classification, potentially applicable to similar datasets.


## Exploratory analysis:

### Platform-Wise Distribution of Ratings:

Initial observations indicate polarizing opinions, with ratings of '1' and '5' being the most common.
A notable disparity exists between the volume of reviews from Google Play Store compared to the App Store, which is considered a limitation of this analysis.

![Platform Wise Distribution of Rating](./images/Platform-Wise_Distribution_of_Ratings-1.png)


## Sentiment Scores

### Findings:

Ratings of '1' exhibit negative sentiment scores for Play Store reviews but show a positive (albeit close to zero) sentiment for App Store reviews.
The variation suggests operating system-based differences in user perceptions, despite a skewed number of App Store reviews.

![Absolute_Sentiment_Scores_by_Rating_and_Source-1.png](./images/Absolute_Sentiment_Scores_by_Rating_and_Source-1.png)
![Absolute_Sentiment_Scores_by_Source.png](./images/Absolute_Sentiment_Scores_by_Source.png)


### Average Sentiment Scores
By normalizing sentiment scores against the total number of reviews per source, an average sentiment score offers a clearer view of overall user feelings towards "Threads".


![Average_Sentiment_Scores_by_Rating_and_Source](./images/(Normalised_by_Count_Method)_Average_Sentiment_Scores_by_Rating_and_Source_1.png)
![Average_Sentiment_Scores_by_Source.png](./images/Average_Sentiment_Scores_by_Source.png)



## Key Word Analysis

### Method: 
Utilized R's built-in algorithms and a manually curated list to exclude common "stop words" and app-specific terms like "twitter," "instagram," etc., to prevent skewed results due to lack of context.


### Findings: 
Words common to both positive and negative reviews include major social platforms, suggesting their use in comparisons.


Key Word Analysis for Positive Reviews (Reviews of rating 4 and above):

![Negative_Reviews_Word_Cloud-1.png](./images/Negative_Reviews_Word_Cloud-1.png)


Key Word Analysis for Negative Reviews (Reviews of rating 2 and below):

![Negative_Reviews_Word_Cloud-1.png](./images/Negative_Reviews_Word_Cloud-1.png)



## Machine Learning Models:

We can use this data to create machine learning models for sentiment classification that can potentially be used to predict values on other such datasets of app reviews. 

I have also taken the precaution of removing the words:  "twitter","instagram","facebook","zuckerberg". I have chosen to exclude these as they are widely used and common to both positive and negative reviews of the app, and could thus impact the predictive power of the model

I followed several steps in the creation of such a model:

(i) Similar to the above word clouds, classifying different levels of 'positive, neutral, and negative.
  
  data$rating >= 4 ~ "positive",
  data$rating == 3 ~ "neutral",
  data$rating <= 2 ~ "negative"

(ii) "Preprocessing" the data for ML analysis: This involves converting text to lowercase, eliminating punctuation, discarding stop words, removing numbers and whitespace. This is all so my model can "read" the data more effectively. 

(iii) Creation of a 'Document Term Matrix (DTM) & calculation of Term Frequency-Inverse Document Frequency (TF-IDF): The DTM is a matrix representation of the dataset where each row corresponds to a document (in this case, a review) and each column represents a unique term across all documents. The values in the matrix indicate the frequency of each term in each document.

TF-IDF is a statistical measure used to evaluate how important a word is. It increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general.

(iv) Pre-paring the above data for ML modeling and splitting the data into 'training' and 'testing' sets: 80% of the total dataset is used to "train" the model while its effectivness in evaluating sentiment is tested on the other 20%.

#### Attempt 1:
Evaluating the model's performance using the test set, I can see it's performance is just above average.Accuracy (0.6074) indicates that 60.74% of all predictions were correct. The low p-value (1.673e-09) indicates that the model is much better than the 'no information rate' (the accuracy achievable by always predicting the most frequent class). However, there is definite room for improvement. A closer look reveals that the model struggles most with 'negative' and 'neutral' classes (due to their relatively low prevalence in the dataset) so I can begin there.

![1st_model.png](./images/1st_model.png)


#### Attempt 2: Adjusting for class sizes via oversampling

Here I will try to adjust for the lower number of negative and neutral reviews:

These are the sizes of each class and the overall average: 

size_negative <- 11522
size_neutral <- 2585
size_positive <- 18803
average_size <- 10970

I will ignore the negative class for now as neutral is drastically more under-represented by targeting halfway between its current size and the average:

target_size_neutral <- round((size_neutral + average_size) / 2)

The neutral class will now be oversampled:
data_neutral_oversampled <- sample_n(data_neutral, target_size_neutral, replace = TRUE)

Unfortunately, this model performs even worse at an overall level than the previous one with an overall decrease in accuracy from from 60.74% to 56.49%. Moreover, the performance on negative and neutral classes, which we were hoping to improve on, actually decreased slightly. 

![2nd_model.png](./images/2nd_model.png)



Another better way of handling class imbalance could be to just simply to two classes, positive and negative.


#### Attempt 3: Using only two classes as opposed to three:

The classes have been redefined to: 
  data$rating >= 4 ~ "positive",
  data$rating <= 3 ~ "negative"

![3rd_model.png](./images/3rd_model.png)

We immediatley see better results. This model, with an accuracy of 66.74% shows a higher overall accuracy than both previous ones (60.74% and 56.49%, respectively).
Specificity and precision are also notably high in this model, especially for identifying positive sentiments and predicting negative sentiments accurately.


However, the sensitivity for negative sentiments is lower. This could demand a more sophisticated treatment of negative class in the data.

One way to achieve this could be via n-grams. An n-gram is a collection of n-succesive words which can add further context and help in model performance. For instance, in previous examples, the word "good" would have been present in many negative reviews being paired with "not" (as in - "not good"). Using n-grams, I can account for this kind of "negation" context when it comes to negative reviews.


I tried a variety of n-gram models including using unigrams, bigrams, and trigrams, just unigrams and bigrams, and finally only bigrams, but attempting to run any of these models took too much of a toll on my local system. (cloud?).

As a result, I have tried to account for this negation problem in the least computationally intensive way I can think of - by handling terms like 'not' by treating it like a single word attached to the next word in the review:

handle_negation <- function(text) {
  text_modified <- gsub("not (\\w+)", "not_\\1", text)
  return(text_modified)
}


![4th_model.png](./images/4th_model.png)


While only slightly better than our previous model (at 66.84%), this one gives us our best overall performance yet. Precision for negative sentiments shows a slight improvement, indicating a minor increase in the accuracy of negative sentiment predictions, but sensitivity for negative sentiments shows a slight decrease from 35.70% to 35.41%, indicating a marginal reduction in the model's ability to correctly identify negative sentiments.









