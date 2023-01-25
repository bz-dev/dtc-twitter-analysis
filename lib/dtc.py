import re
from collections import defaultdict
from typing import Union, Set, List
from loguru import logger
import pandas
from gensim import corpora


@logger.catch()
def tokenizer(text: str, stopwords: Union[Set[str], List[str]]):
    pattern = re.compile("^[a-z0-9#].*")
    return [
        word.strip()
        .strip(".")
        .strip('"')
        .strip("'")
        .strip("’")
        .strip("?")
        .strip("!")
        .strip(":")
        .strip("[")
        .strip("]")
        .strip("-")
        .strip("(")
        .strip(")")
        .replace("’", "'")
        for word in re.split('[,; ":\n\(\)]|\.{2,}', text.lower())
        if len(word) > 0
        and word not in stopwords
        and not word.startswith("@")
        and not word.startswith("http")
        and not word.isnumeric()
        and pattern.match(word)
    ]


def get_topics(text, model, dictionary, stopwords):
    tokens = tokenizer(text, stopwords)
    corpus = dictionary.doc2bow(tokens)
    dist = model.get_document_topics(corpus)
    topics = [None, None, None, None, None, None, None, None]
    for item in dist:
        for i in range(9):
            if item[0] == i:
                topics[i] = item[1]
                break
    top = sorted(dist, key=lambda item: item[1], reverse=True)[0][0] + 1
    result = [top] + topics
    return pandas.Series(result)


def nlp_preprocess(texts: List[str], stopwords: Union[Set[str], List[str]]):
    tokens_list = [tokenizer(text, stopwords) for text in texts]
    frequency = defaultdict(int)
    for tokens in tokens_list:
        for token in tokens:
            frequency[token] += 1
    processed_corpus = [
        [token for token in tokens if frequency[token] > 1] for tokens in tokens_list
    ]
    dictionary = corpora.Dictionary(processed_corpus)
    corpus = [dictionary.doc2bow(doc) for doc in processed_corpus]
    return corpus, dictionary, tokens_list
