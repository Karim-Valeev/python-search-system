import os

import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

from server.settings import BASE_DIR

STOPWORDS = set(stopwords.words('english'))


def get_link_to_filename_mapping() -> dict:
    path = BASE_DIR.parent.joinpath("index.txt")
    mapping = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            file, url = line.split(' - ')
            url = url[:-1]
            mapping[file] = url
    return mapping


def get_lemmas_from_query(query: set, tokenizer, morph_analyzer) -> list:
    lemmas = []
    for token in set(tokenizer.tokenize(query)):
        morph = morph_analyzer.parse(token)
        if token in STOPWORDS:
            continue
        lemmas.append(morph[0].normal_form)
    return lemmas


def get_query_vector(query_lemmas, all_lemmas):
    vector = {lemma: 0 for lemma in all_lemmas}
    for lemma in query_lemmas:
        vector[lemma] = 1
    return vector


def get_all_lemmas():
    path = BASE_DIR.parent.joinpath("inverted_index.txt")
    lemmas = []
    with open(path, 'r') as f:
        for line in f.readlines():
            lemmas.append(line.split(maxsplit=1)[0])
    return lemmas


def get_page_links():
    path = BASE_DIR.parent.joinpath("index.txt")
    pages = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            file, url = line.split(' - ')
            url = url[:-1]
            file = file.strip('.html')
            file = file.strip('site_')
            file = f'tf-idf_{file}.txt'
            pages[file] = url
    return pages


def vector_normalize(vector):
    return sum([c ** 2 for c in vector]) ** 0.5


def calculate_cosine_similarity(query_vector, page_vector):
    dot = sum(q * p for q, p in zip(query_vector, page_vector))
    if dot:
        return dot / (vector_normalize(query_vector) * vector_normalize(page_vector))
    return 0


def get_similarities(query_vector: list, tf_idf_matrix, page_links):
    similarities = {}
    for page, lemma_tf_idf in tf_idf_matrix.items():
        page_vector = list(lemma_tf_idf.values())
        similarity = calculate_cosine_similarity(query_vector, page_vector)
        if similarity:
            similarities[page] = similarity
    return sorted(
        [(page_links[x[0]], x[1]) for x in similarities.items() if x[1] > 0.0],
        key=lambda x: x[1],
        reverse=True,
    )


def get_tf_idf_matrix(all_lemmas):
    matrix = {}
    # path = BASE_DIR.parent.joinpath("lemmas_tf_idf")
    path = "/home/pain/Desktop/INFO_SEARCH/python-search-system/lemmas_tf-idf"
    for root, _, files in os.walk(path):
        for file in files:
            with open(os.path.join(root, file)) as f:
                matrix[file] = {lemma: 0.0 for lemma in all_lemmas}
                for line in f.readlines():
                    lemma, _, tf_idf = line.split(maxsplit=2)
                    matrix[file][lemma] = float(tf_idf)
    return matrix


def search_file_links(query):
    tokenizer = WordPunctTokenizer()
    morph_analyzer = pymorphy2.MorphAnalyzer()
    all_lemmas = get_all_lemmas()
    tf_idf_matrix = get_tf_idf_matrix(all_lemmas)
    page_links = get_page_links()

    query_lemmas = get_lemmas_from_query(query, tokenizer, morph_analyzer)
    query_vector = get_query_vector(query_lemmas, all_lemmas)
    similarities = get_similarities(list(query_vector.values()), tf_idf_matrix, page_links)
    links = [x[0] for x in similarities]
    return links


if __name__ == '__main__':
    print(search_file_links('Pegasus'))


