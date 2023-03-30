import ast
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer


STOPWORDS = stopwords.words('english')


def get_tokens(s: str) -> list:
    tokenizer = RegexpTokenizer('[A-Za-z]+')
    clean_words = tokenizer.tokenize(s)
    clean_words = [word.lower() for word in clean_words if word != '']
    clean_words = [word for word in clean_words if word not in STOPWORDS]
    return clean_words


def get_lemmas(tokens: set) -> dict:
    pymorphy2_analyzer = MorphAnalyzer()
    lemmas = {}
    for token in tokens:
        if re.match(r'[A-Za-z]', token):
            lemma = pymorphy2_analyzer.parse(token)[0].normal_form
            if lemmas.get(lemma):
                lemmas[lemma].append(token)
            else:
                lemmas[lemma] = [token]
    return lemmas



def search_files(query):
    query_lemmas = get_lemmas(query)
    query_vector = get_query_vector(query_lemmas)
    return get_similarities(list(query_vector.values()))


