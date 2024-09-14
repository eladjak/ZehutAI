from sentence_transformers import SentenceTransformer


def compare_sentences(sentences):
    """
    Takes in two sentences, returns their cosine similarity using
    sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    :param sentences: list [str, str]
        list of 2 sentences to be compared
    :return: float
        cosine similarity
    """
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    embeddings = model.encode(sentences)
    similarities = model.similarity(embeddings, embeddings)
    return float(similarities[0, 1])


sentences = ["This is an example sentence", "זה משפט כדוגמה "]

sentences2 = ["This is an example sentence", "לעשות משפט וחסד"]

sentences3 = ["This is an example sentence", "תכתוב פה מה שאני כותב"]

sentences4 = ["This is an example sentence", "אני רוצה לשתות מים"]

for s in [sentences, sentences2, sentences3, sentences4]:
    print(s[0], s[1][::-1], ': ', str(compare_sentences(s)))
