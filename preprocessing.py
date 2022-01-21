import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def hashtags(message):
    hash = re.findall("#(\\w+)", message)
    return hash


def removeHashtags(message):
    return re.sub("(\\s*#(\\w+)\\s*)+", " ", message)


def mentions(message):
    ment = re.findall("@(\\w+)", message)
    return ment


def removeMentions(message):
    return re.sub("(\\s*@(\\w+)\\s*)+", " ", message)


def urls(message):
    url = re.findall('https?://((?:[-\\w.]|(%[\\da-fA-F]{2}))/?)+(#\\w+(=\\w+)*)*', message)
    return url


def removeUrls(message):
    return re.sub('(\\s*https?://((?:[-\\w.]|(%[\\da-fA-F]{2}))/?)+(#\\w+(=\\w+)*)*\\s*)+', " ", message)


def clearText(message):
    tokens = nltk.word_tokenize(message)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return words


def preprocess(url, output='data/preprocessedData.csv'):
    data = pd.read_csv(url)
    data["urlsCount"] = data["Tweet"].apply(urls).apply(len)
    data["Tweet"] = data["Tweet"].apply(removeUrls)
    data["hashtags"] = data["Tweet"].apply(hashtags)
    data["hashtagsCount"] = data["hashtags"].apply(len)
    data["mentions"] = data["Tweet"].apply(mentions)
    data["mentionsCount"] = data["mentions"].apply(len)
    data["Tweet"] = data["Tweet"].apply(removeMentions).apply(removeHashtags)
    data["sentencesCount"] = data["Tweet"].apply(nltk.sent_tokenize).apply(len)
    data["charsCount"] = data["Tweet"].apply(len)
    data["numbersToChars"] = data["Tweet"].apply(lambda x: len([c for c in x if x.isnumeric()]))
    data["wordsCount"] = data["Tweet"].apply(clearText).apply(len)
    data["hashtagsPercentage"] = data["hashtagsCount"] / (data["wordsCount"] + data["hashtagsCount"])
    data["urlsPercentage"] = data["urlsCount"] / (data["wordsCount"] + data["urlsCount"])
    data["mentionsPercentage"] = data["mentionsCount"] / (data["wordsCount"] + data["mentionsCount"])
    vectorizer = CountVectorizer(tokenizer=clearText, token_pattern=None, max_features=int(1e2))
    x = vectorizer.fit_transform(data["Tweet"])
    x=pd.DataFrame.sparse.from_spmatrix(x)
    data=pd.concat([data,x],axis=1)
    data = data.drop(['Tweet','hashtags','mentions','Id'], axis=1)
    place = LabelEncoder()
    data["location"] = place.fit_transform(data["location"])
    data["Type"] = data["Type"].apply(lambda x: x == "Quality")
    data = data.fillna(0)
    data.to_csv(output, index=False)
