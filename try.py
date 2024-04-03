import nltk
import spacy
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
 
lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')

def get_word_synsets(word):
    synsets = wn.synsets(word)
    return synsets

def calculate_synset_similarity(synset1, synset2):
    if synset1 == synset2:
        return 1.0 
    else:
        hypernyms1 = synset1.hypernyms()
        hypernyms2 = synset2.hypernyms()

        ns1 = set()
        ns2 = set()

        for hypernym in hypernyms1:
            ns1.update(hypernym.hyponyms())

        for hypernym in hypernyms2:
            ns2.update(hypernym.hyponyms())

        # Convert to set of names
        ns1 = {s.name() for s in ns1}
        ns2 = {s.name() for s in ns2}

        intersection_size = len(ns1.intersection(ns2))
        min_size = min(len(ns1), len(ns2))
        if min_size == 0:
            return 0.0
        else:
            return intersection_size / min_size


def align_sentences(sent1, sent2, threshold=0.5):
    alignments = []
    tokens1 = join_n_grams(sent1, 1)
    tokens1.extend(join_n_grams(sent1, 2))
    tokens1.extend(join_n_grams(sent1, 3))
    tokens2 = join_n_grams(sent2, 1)
    tokens2.extend(join_n_grams(sent2, 2))
    tokens2.extend(join_n_grams(sent2, 3))

    for token1 in tokens1:
        if token1.lower() in nltk.corpus.stopwords.words('english'):
            continue

        synsets1 = get_word_synsets(token1)

        max_similarity = 0
        aligned_token = None

        for token2 in tokens2:
            if token2.lower() in nltk.corpus.stopwords.words('english'):
                continue

            synsets2 = get_word_synsets(token2)

            for synset1 in synsets1:
                for synset2 in synsets2:
                    similarity = calculate_synset_similarity(synset1, synset2)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        aligned_token = token2

        if max_similarity >= threshold:
            alignments.append((token1, aligned_token, max_similarity))

    return alignments

def join_n_grams(sentence,n):
    tokens = word_tokenize(sentence)
    n_grams = list(nltk.ngrams(tokens, n))
    if n == 1:
        n_grams = [gram[0] for gram in n_grams]
    else:
        n_grams = ['_'.join(gram) for gram in n_grams]
    return n_grams

sentence1 = "The old guy kicked the bucket at the age of 70."
sentence2 = "The old guy died at the age of seventy."

sent1 = nlp(sentence1)
sent2 = nlp(sentence2)
 
lemmatized_tokens1 = [token.lemma_ for token in sent1]
lemmatized_tokens2 = [token.lemma_ for token in sent2]
 
lemmatized_sentence1 = ' '.join(lemmatized_tokens1)
lemmatized_sentence2 = ' '.join(lemmatized_tokens2)

alignments = align_sentences(lemmatized_sentence1, lemmatized_sentence2)

print(alignments)