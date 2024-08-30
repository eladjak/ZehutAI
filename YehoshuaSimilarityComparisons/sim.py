from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
from scipy.spatial import distance
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import transformers
from transformers import BertTokenizer, RobertaTokenizer
import numpy as np
import tensorflow as tf
import matplotlib



class Similarity:
    def __init__(self):
        self.methods = [self.methodNLTK]
        self.nnModels = []
        self.texts = []
        self.methods = []


    def add_texts(self):
        return None


    def compareMethods(self, text1=None, text2=None):
        for method in self.methods:
            result = method()
            print(result)

    def methodNLTK(self, data=None):
        # Sample data
        data = ["The movie is awesome. It was a good thriller",
                "We are learning NLP throughg GeeksforGeeks",
                "The baby learned to walk in the 5th month itself"]

        # Tokenizing the data
        tokenized_data = [word_tokenize(document.lower()) for document in data]

        # Creating TaggedDocument objects
        tagged_data = [TaggedDocument(words=words, tags=[str(idx)])
                       for idx, words in enumerate(tokenized_data)]

        # Training the Doc2Vec model
        model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=1000)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count,
                    epochs=model.epochs)

        # Infer vector for a new document
        new_document = "The baby was laughing and palying"
        print('Original Document:', new_document)

        inferred_vector = model.infer_vector(word_tokenize(new_document.lower()))

        # Find most similar documents
        similar_documents = model.dv.most_similar(
            [inferred_vector], topn=len(model.dv))

        # Print the most similar documents
        for index, score in similar_documents:
            print(f"Document {index}: Similarity Score: {score}")
            print(f"Document Text: {data[int(index)]}")
            print()
        return None

    def methodScikitlearn(self):

        data = ["The movie is awesome. It was a good thriller",
                "We are learning NLP throughg GeeksforGeeks",
                "The baby learned to walk in the 5th month itself"]

        text2 = "The baby was laughing and palying"

        # Convert the texts into TF-IDF vectors
        vectorizer = TfidfVectorizer()
        print('text2: ' + text2)
        for text1 in data:
            vectors = vectorizer.fit_transform([text1, text2])
            # Calculate the cosine similarity between the vectors
            similarity = cosine_similarity(vectors)
            print(text1 + ' ' + str(similarity[1, 0]))


    def methodNNEmbeddings(self, tokenizer, model, model_weights):
        """
        Finds the distances between the user input and the source texts
        :param tokenizer: The tokenizer for the model (e.g. BertTokenizer)
        :param model: the model (e.g. BertModel)
        :param model_weights: str
            The pretrained weights for the model (e.g. 'bert-base-uncased)
        :return: list[Tuple (float, str, str)]
            List of Tuples holding information regarding the user prompt and each text:
             1. The distance from the embeddings of the user prompt and each text
             2. The text
             3. The user prompt
        """
        # Tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)

        # Load the BERT model
        model = transformers.BertModel.from_pretrained('bert-base-uncased')

        # Tokenize and encode the texts
        data = ["The movie is awesome. It was a good thriller",
                "We are learning NLP throughg GeeksforGeeks",
                "The baby learned to walk in the 5th month itself"]

        user_prompt = "The baby was laughing and palying"
        tokenized_user_prompt = tokenizer(user_prompt, return_tensors="pt", padding="max_length", return_attention_mask=True)

        texts_list = []
        for target_text in data:
            tokenized_target_text = tokenizer(target_text, return_tensors="pt", padding="max_length", return_attention_mask=True)

            embedding1 = model(tokenized_target_text['input_ids'], attention_mask=tokenized_target_text['attention_mask'])[0].detach().numpy()[0, :, 0]
            embedding2 = model(tokenized_user_prompt['input_ids'], attention_mask=tokenized_user_prompt['attention_mask'])[0].detach().numpy()[0, :, 0]

            # Calculate the cosine similarity between the embeddings
            similarity = np.dot(embedding1.T, embedding2) / (np.linalg.norm(embedding1.T) * np.linalg.norm(embedding2))
            texts_list.append((similarity, target_text, user_prompt))
        return texts_list

    def runNNModels(self):
        for model_objects in self.nnModels:
            tokenizer, model, model_weights = model_objects







    def methodBert(self):
        # Tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)

        # Load the BERT model
        model = transformers.BertModel.from_pretrained('bert-base-uncased')

        # Tokenize and encode the texts
        data = ["The movie is awesome. It was a good thriller",
                "We are learning NLP throughg GeeksforGeeks",
                "The baby learned to walk in the 5th month itself"]

        text2 = "The baby was laughing and palying"
        tokenized2 = tokenizer(text2, return_tensors="pt", padding="max_length", return_attention_mask=True)

        for text1 in data:
            tokenized1 = tokenizer(text1, return_tensors="pt", padding="max_length", return_attention_mask=True)

            # encoding1 = tokenizer.encode(text1[''], max_length=512, add_special_tokens=True, padding=True)
            # encoding2 = tokenizer.encode(text2, max_length=512, add_special_tokens=True, padding=True)
            # print(text1, text2)
            # tokenized1 = tokenizer.tokenize(encoding1)
            # print('tokenized1 ' + str(tokenized1))

            embedding1 = model(tokenized1['input_ids'], attention_mask=tokenized1['attention_mask'])[0].detach().numpy()[0, :, 0]
            embedding2 = model(tokenized2['input_ids'], attention_mask=tokenized2['attention_mask'])[0].detach().numpy()[0, :, 0]

            # Calculate the cosine similarity between the embeddings
            similarity = np.dot(embedding1.T, embedding2) / (np.linalg.norm(embedding1.T) * np.linalg.norm(embedding2))
            print(similarity, text1, ' | ', text2)

    def methodRoBERTa(self):
        # Tokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', clean_up_tokenization_spaces=True)

        # Load the BERT model
        model = transformers.RobertaModel.from_pretrained('roberta-base')

        # Tokenize and encode the texts
        data = ["The movie is awesome. It was a good thriller",
                "We are learning NLP throughg GeeksforGeeks",
                "The baby learned to walk in the 5th month itself"]

        text2 = "The baby was laughing and palying"
        tokenized2 = tokenizer(text2, return_tensors="pt", padding="max_length", return_attention_mask=True)

        for text1 in data:
            tokenized1 = tokenizer(text1, return_tensors="pt", padding="max_length", return_attention_mask=True)

            # encoding1 = tokenizer.encode(text1[''], max_length=512, add_special_tokens=True, padding=True)
            # encoding2 = tokenizer.encode(text2, max_length=512, add_special_tokens=True, padding=True)
            # print(text1, text2)
            # tokenized1 = tokenizer.tokenize(encoding1)
            # print('tokenized1 ' + str(tokenized1))

            embedding1 = model(tokenized1['input_ids'], attention_mask=tokenized1['attention_mask'])[0].detach().numpy()[0, :, 0]
            embedding2 = model(tokenized2['input_ids'], attention_mask=tokenized2['attention_mask'])[0].detach().numpy()[0, :, 0]
            # Calculate the cosine similarity between the embeddings
            similarity = np.dot(embedding1.T, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            print(similarity, text1, ' | ', text2)

    # def methodSBERT(self):



similarity = Similarity()
# similarity.methodNLTK()
# similarity.methodScikitlearn()
# similarity.methodBert()
# similarity.methodRoBERTa()

