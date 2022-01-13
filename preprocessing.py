import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder


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
    words = [w for w in words if w not in stop_words]
    return words


def preprocess(url):
    data = pd.read_csv(url)
    data["urlsCount"] = data["Tweet"].apply(urls).apply(len)
    data["Tweet"] = data["Tweet"].apply(removeUrls)
    data["hashtagsCount"] = data["Tweet"].apply(hashtags).apply(len)
    data["mentionsCount"] = data["Tweet"].apply(mentions).apply(len)
    data["Tweet"] = data["Tweet"].apply(removeMentions).apply(removeHashtags)
    data["sentencesCount"] = data["Tweet"].apply(nltk.sent_tokenize).apply(len)
    data["charsCount"] = data["Tweet"].apply(len)
    data["numbersToChars"] = data["Tweet"].apply(lambda x: len([c for c in x if x.isnumeric()]))
    data["wordsCount"] = data["Tweet"].apply(clearText).apply(len)
    data["hashtagsPercentage"] = data["hashtagsCount"] / (data["wordsCount"] + data["hashtagsCount"])
    data["urlsPercentage"] = data["urlsCount"] / (data["wordsCount"] + data["urlsCount"])
    data["mentionsPercentage"] = data["mentionsCount"] / (data["wordsCount"] + data["mentionsCount"])
    data = data.drop('Tweet', axis=1)
    data = data.drop('Id', axis=1)
    place = LabelEncoder()
    data["location"] = place.fit_transform(data["location"])
    data["Type"] = data["Type"].apply(lambda x: x == "Quality")
    data=data.fillna(0)
    data.to_csv('data/preprocessedData.csv', index=False)
