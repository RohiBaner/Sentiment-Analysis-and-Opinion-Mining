# Import all necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer

# Read the training data for sentiment analysis
# Stanford's Sentiment140 dataset has been used as training data
cols = ['sentiment','num','date','query','id','tweet']
data = pd.read_csv('train.csv',names=cols, encoding='cp1252')
# Drop any duplicates and unnecessary columns
data.dropna(inplace=True)
data.drop(['num','date','query','id'],axis=1,inplace=True)
# Reassign the positive sentiment to be equal to '1'
data.loc[data.sentiment == 4, 'sentiment'] = 1

# Preparing data for cleaning
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9_]+' # Remove '@' from tweets
pat2 = r'https?://[^ ]+' # Remove http links from tweets
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+' # Remove websites from the tweets
# convert negative apostrophe words into two separate words
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
rt = re.compile('(\s*)rt(\s*)') # Remove 'RT' (retweet) words from tweets

# Function to clean the tweets wrt to the above mentioned cleaning aspects
def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml') # Perform HTML encoding
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower() # convert everything to lower case
    lower_case = rt.sub('', lower_case)
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces
    # Tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()

# Clean the training set data wrt to the above cleaning function
print("Cleaning and parsing the training tweets...")
clean_tweets = []
for i in data.tweet:
    clean_tweets.append(tweet_cleaner(i))
print("Done!\n")
# Get the final cleaned tarining dataset
clean_df = pd.DataFrame(clean_tweets, columns=['tweet'])
clean_df['sentiment'] = data.sentiment

# Read the test data
test = pd.read_csv('test.csv', encoding='cp1252')
# Clean the test data wrt to the above cleaning function as well
print("Cleaning and parsing the testing tweets...")
clean_test = []
for i in test.Tweet:
    clean_test.append(tweet_cleaner(i))
print("Done!\n")
# Get the final dataframe containing the cleaned test data
test1 = pd.DataFrame(clean_test, columns=['tweet'])

# Vectorize the tweets for both training and testing datasets
vect1 = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(clean_df['tweet'].values)
X_vect = vect1.transform(clean_df['tweet'].values)
test_vect = vect1.transform(test1['tweet'].values)
# Train the dataset with Logistic Regression
model = LogisticRegression()
model.fit(X_vect, clean_df['sentiment'].values)
predictions = model.predict(test_vect) # Predict the results
print(predictions.mean(),"of the total test data contains tweets of positive sentiments!")

# Obtain the CSV file in the required format
test.drop(['Date','User','v','a'],axis=1, inplace=True)
test.reset_index(inplace=True)
test.columns = ['ID', 'Contents'] # Change column names to match the format specified
pred_df = pd.DataFrame(predictions, columns =['Result']) # Convert predictions into a dataframe
result = pd.concat([test, pred_df], axis=1, sort=False) # Merge the two dataframes
result.to_csv('output.csv', sep=',', header=True, index=False, encoding='cp1252')
