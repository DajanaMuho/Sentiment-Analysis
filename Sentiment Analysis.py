import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
                                    #Data Gathering 
#Read csv file
tweets = pd.read_csv('tweets_raw.csv')

tweets.shape # (202645, 8)
pd.options.display.max_columns = None
pd.options.display.max_colwidth = -1
tweets.columns
tweets.head()

                                    #Data Preparation
tweets.isnull().any()
# Fill missing values of Location
location = tweets['Location']
location.isnull().sum(axis=0) #47522 nan values
location.unique()
tweets['Location'].fillna('unknown', inplace=True)
# Drop uninformative columns
tweets = tweets.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
# Remove duplicates
tweets = tweets.drop_duplicates()
tweets.shape # (187052, 6), remove 15593 duplicated rows
#Data type conversions
tweets.dtypes
tweets['Created at'] = pd.to_datetime(tweets['Created at']) #convert to datetime

                                    #Exploratory Data Analysis
#Investigation on data
corr = tweets.corr()
corr
sns.heatmap(corr)
sns.regplot(x="Favorites", y="Retweet-Count", data=tweets)
sns.distplot(tweets["Created at"].dt.month, bins=24, axlabel="Monthly distribution of tweets")
#The most popular tweets
popularTweets = tweets.sort_values(by=["Favorites","Retweet-Count", ], axis=0, ascending=False).head()
plt.xticks(rotation=15)
sns.countplot(x="Location", hue="Favorites", data=popularTweets)
sns.countplot(x="Location", hue="Retweet-Count", data=popularTweets)

                                #Feature Engineering
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

def clearWord(content):
    #remove links
    content = re.sub(r"http\S+|www\S+|https\S+", '', content, flags=re.MULTILINE)
    #remove mentions and hashtag
    content = re.sub(r'\@\w+|\#','', content) 
    # tokenize the words, by splitting into small units
    word = word_tokenize(content) 
    #remove stop words that doesn't add much meaning to a sentence
    word = [c for c in word if not c.lower() in stopwords.words('english')] 
    #lemmatize the words, return the base of of words known as lemmalemmatize the words, return the base of of words known as lemma
    lemmatizer = WordNetLemmatizer()
    word = [lemmatizer.lemmatize(w, pos='a') for w in word] 
    # remove non-alphabetic characters and keep the words contains three or more letters
    tokenized = [c for c in word if c.isalpha() and len(c) > 2]
    return word;

tweets['Processed'] = tweets["Content"].str.lower().apply(clearWord)
    
tweets.to_csv('tweetsProcessed.csv', index=False)
#tweets = pd.read_csv('tweetsProcessed.csv')      

import pycountry

def getCountryCode(location):
    if pycountry.countries.get(name= location):
        return pycountry.countries.get(name = location).alpha_2
    try:
        pycountry.subdivisions.lookup(location)
        return pycountry.subdivisions.lookup(location).country_code
    except:
        return "unknown"
tweets["Country"] = tweets["Location"].apply(getCountryCode)

tweetsFilterByCountries = tweets[tweets["Country"] != 'unknown']
popularTweets = tweetsFilterByCountries.sort_values(by=["Favorites","Retweet-Count", ], axis=0, ascending=False).head(n=5)
popularTweets = popularTweets.drop(['Processed'], axis=1).drop_duplicates(subset="Location", keep="first")
sns.countplot(x="Location", hue="Favorites", data=popularTweets)
sns.countplot(x="Location", hue="Retweet-Count", data=popularTweets)

                             # Sentiment Analysis
from textblob import TextBlob

def findPolarity(word): #How positive or negative a word is
    return TextBlob(' '.join(word)).sentiment.polarity

def findSubjectivity(word):#How objective or subjective a word is
    return TextBlob(' '.join(word)).sentiment.subjectivity

def findLabelPolarity(polarity):
    if polarity > 0:
        return "Positive"
    if polarity == 0:
        return "Neutral"
    if polarity < 0:
        return "Negative"

tweets['Polarity'] = tweets['Processed'].apply(findPolarity)
tweets['Subjectivity'] = tweets['Processed'].apply(findSubjectivity)
tweets['Polarity Label'] = tweets['Polarity'].apply(findLabelPolarity)
tweets[["Polarity","Subjectivity", "Polarity Label"]]

tweets["Polarity Label"].value_counts()

sns.countplot(tweets['Polarity Label'])
sns.scatterplot(x='Subjectivity', y='Polarity', data=tweets)
#Positive
tweets.sort_values(by=["Polarity","Favorites","Retweet-Count", ], axis=0, ascending=[False, False, False])[["Content","Retweet-Count","Favorites","Polarity Label"]].head()
#Negative
tweets.sort_values(by=["Polarity","Favorites","Retweet-Count", ], axis=0, ascending=[True, False, False])[["Content","Retweet-Count","Favorites","Polarity Label"]].head()

#Word Cloud for negative and positive tweets
stopWords = ["online","class","course","learning","learn",
"teach","teaching","distance","distancelearning","education",
"teacher","student","grade","classes","computer","onlineeducation", "onlinelearning", "school", "students","class","virtual","eschool","virtuallearning", "educated", "educates", "teaches", "studies", "study", "semester", "elearning","teachers", "lecturer", "lecture", "amp","academic", "admission", "academician", "account", "action",
"add", "app", "announcement", "application", "adult", "classroom", "system", "video", "essay", "homework","work","assignment","paper","get", "math", "project", "science", "physics", "lesson","courses", "assignments", "know", "instruction","email", "discussion","home", "college","exam""use","fall","term","proposal","one","review",
"proposal", "calculus", "search", "research", "algebra"]
from wordcloud import WordCloud
def plotWordCloud(all_words):
    wc = WordCloud(width = 500, height = 500, min_font_size = 10, max_words=2000, background_color ='white', stopwords= stopWords)
    polarity = wc.generate(all_words)                  
    plt.figure(figsize = (10, 10), facecolor = None) 
    plt.imshow(polarity) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show() 

#common words in the negative tweets
mnegative=tweets['Polarity'].min()
mnegative
mostNegative = tweets[tweets["Polarity"] == mnegative]
mostNegative.Processed = mostNegative["Processed"].apply(lambda txt: ' '.join(txt))
plotWordCloud(' '.join(mostNegative["Processed"]))

#common words in the positive tweets
mpositive=tweets['Polarity'].max()
mpositive
mostPositive = tweets[tweets["Polarity"] == mpositive]
mostPositive.Processed = mostPositive["Processed"].apply(lambda txt: ' '.join(txt))
plotWordCloud(" ".join(mostPositive["Processed"])) 


# Polarity by country    
topCountries = tweetsFilterByCountries["Country"].value_counts().head()
index = tweets["Country"].isin(topCountries.index[:10]).values
ctrDf = tweets.iloc[index,:]
sns.countplot(x="Country", hue="Polarity Label", data=ctrDf)


                #Machine Learning
# Encode the labels
#“Positive” = 2
#“Neutral” = 1
#“Negative” = 0
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

tweets["Label_enc"] = le.fit_transform(tweets["Polarity Label"])
tweets[["Label_enc"]].head()

# Select the features and the target
X = tweets['Processed']
y = tweets["Label_enc"]

#use the stratify parameter of train_test_split since our data is unbalanced.
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=34, stratify=y)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
Xtrain = vectorizer.fit_transform(Xtrain.apply(lambda x: ' '.join(x)))
Xtest = vectorizer.transform(Xtest.apply(lambda x: ' '.join(x)))

# Naive Bayes classifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
nb = BernoulliNB()
nb.fit(Xtrain, ytrain)
yPred = nb.predict(Xtest)
print("Accuracy:",  accuracy_score(yPred, ytest)) #0.21127475876079227
print("Test Error Rate:",  1- accuracy_score(yPred, ytest)) #0.7887252412392077



#RandomForest classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100, random_state = 1)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)
print("Accuracy:",  accuracy_score(ypred, ytest)) #0.863810109326134
print("Test Error Rate:",  1- accuracy_score(ypred, ytest)) #0.136189890673866
