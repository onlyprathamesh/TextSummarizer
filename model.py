import pickle
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize into sentences
    sentences = nltk.sent_tokenize(text)
    # Remove punctuation and stop words, and lowercase all words
    table = str.maketrans('', '', string.punctuation)
    stopwords = set(nltk.corpus.stopwords.words('english'))
    preprocessed_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words = [word.lower() for word in words]
        words = [word.translate(table) for word in words]
        words = [word for word in words if word.isalpha()]
        words = [word for word in words if word not in stopwords]
        preprocessed_sentences.append(' '.join(words))
    return preprocessed_sentences

def text_summarization(text, num_sentences=1):
    preprocessed_sentences = preprocess_text(text)
    # Convert preprocessed sentences into a single string
    preprocessed_text = ' '.join(preprocessed_sentences)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
    # Return the summarized text
    return preprocessed_text


with open('text_summarization_model.pkl', 'wb') as f:
    pickle.dump(text_summarization, f)
