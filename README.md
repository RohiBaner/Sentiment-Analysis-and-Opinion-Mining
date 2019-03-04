
# Sentiment Analysis and Opinion Mining (Python 3) #
Generally speaking, sentiment analysis aims to determine the attitude of a speaker, writer, or other subject with respect to some topic or the overall contextual polarity or emotional reaction to a document, interaction, or event. The attitude may be a judgment or evaluation, affective state, or the intended emotional communication.
In this solution to the given problem, sentiments of tweets pertaining to genetic technology was analysed.


## 1. Getting Started ##
This solution of Sentiment Analysis and Opinion Mining (Python3) initially trains on Stanford’s Sentiment140 dataset. The trained model can then be applied to tweets related to genetic technology. I have chosen the Sentiment140 dataset and not the IMDB dataset as this dataset pertains to the language corpus of tweets. It is also evenly balanced in nature.

### 1.1. Data Description ###
As mentioned above, the training dataset pertains to the Stanford Sentiment140 dataset. It contains 6 attributes (including the polarity/sentiment of the tweet and the tweet itself) and about 1.6 million entries. The dataset is well balanced and clean.
For the testing data, I have taken 100 tweets pertaining to genetic technology. As crawling the data would have taken too much time, I have used an online tool (Twitter Sentiment Visualization App) to get the data by providing it some keywords. I have used the keywords “gene editing”, “transgene” and “genetic engineering”.

### 1.2. Prerequisites ###
The following libraries are required to run the code. All of them can be downloaded via conda or pip. Please make sure that the [train.csv](https://drive.google.com/file/d/1b7RY4FK9P2N_-4WyiUp1CrohyJJlgCOH/view) and [test.csv](https://drive.google.com/file/d/1vlTqBZzGL-m9mPzvUDBJFpFo-hU5-qC8/view) files are downloaded and kept in the same folder as the code before running.
1. pandas
2. sklearn (LogisticRegression, CountVectorizer)
3. pprint
4. re
5. BeautifulSoup
6. nltk

### 1.3. Running the Code ###
The code can be run using terminal or any Python IDE (ex. Spyder). To run via terminal by navigating to the required directory and running the following command (please ensure that both train and test datasets are kept in the same directory as the code):
**python Q5_code_sentiment.py**

## 2. Methodology ##
The following steps are followed to train the Sentiment140 dataset and then test the model to obtain the sentiments for the test data.
1. Import all the necessary libraries
2. Read the Sentiment140 dataset and change the label for positive sentiment to ‘1’ (instead of the given ‘4’)
   - Clean the training data tweets via a function (this function will be called to clean the test data tweets as well): o Perform HTML encoding with BeautifulSoup
   - Remove “https://”, “www”, “@” and “rt” from the tweets to get cleaner results
   - Separate out negative conjunct words (aren’t, couldn’t, etc) as two separate words
   - Remove punctuations
   - Convert everything to lower case
   - Remove any unnecessary spaces
   - Tokenize all words
3. Clean the test data tweets via the above steps as well
4. Vectorize the tweets for both training and test data
5. Fit the model on training data (where sentiment values are provided)
6. Test the model for the collected test dataset (0: negative sentiment; 1: positive sentiment)
7. Arrange results in required output format

For this question I have used Logistic Regression as the model. It provided an AUC(ROC) score of about 82% when I validated it with part of the training data. It provided better results than XGBClassifier (70% AUC-ROC). DecisionTree and RandomForest took too long to run. Hence, I stuck with Logistic Regression for my solution.

## 3. Results and Discussion ##
The model predicted 70% of the results to be of positive sentiment. The Twitter Sentiment Visualization App that I used to gather the tweets also gave similar results. Hence the model is in sync with the accurate output. Most people (on Twitter, at least) are eager to know more about the nuances of genetic technology and the positive changes it can bring to medical treatment.

## 4. Author ##
BANERJEE, Rohini - HKUST Student ID: 20543577

## 5. References ##
1. https://towardsdatascience.com/another-twitter-sentiment-analysis-bb5b01ebad90
2. https://github.com/ehsansade/Sentiment-Analysis/blob/master/iPhone%20X%20-%20Sentiment%20Analysis.ipynb
3. https://www.csc2.ncsu.edu/faculty/healey/tweet_viz/tweet_app/
4. https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-2-333514854913
5. http://help.sentiment140.com/for-students

