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
    similarity = synset1.path_similarity(synset2)
    return similarity

def align_sentences(sent1, sent2, threshold=0.3):
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
        
        if token1 in ['.', ',', '?', '!', ':', ';']:
            continue

        synsets1 = get_word_synsets(token1)

        max_similarity = 0

        for token2 in tokens2:
            if token2.lower() in nltk.corpus.stopwords.words('english'):
                continue

            if token2 in ['.', ',', '?', '!', ':', ';']:
                continue

            synsets2 = get_word_synsets(token2)

            for synset1 in synsets1:
                for synset2 in synsets2:
                    similarity = calculate_synset_similarity(synset1, synset2)
                    if similarity is not None and similarity > max_similarity:
                        max_similarity = similarity
            if max_similarity >= threshold:
                alignments.append((token1, token2, max_similarity))

    return alignments, tokens1, tokens2

def join_n_grams(sentence,n):
    tokens = word_tokenize(sentence)
    n_grams = list(nltk.ngrams(tokens, n))
    if n == 1:
        n_grams = [gram[0] for gram in n_grams]
    else:
        n_grams = ['_'.join(gram) for gram in n_grams]
    return n_grams

sentence1 = "The tree provided shade in the park."
sentence2 = "The oak offered shelter in the forest."

sent1 = nlp(sentence1)
sent2 = nlp(sentence2)
 
lemmatized_tokens1 = [token.lemma_ for token in sent1]
lemmatized_tokens2 = [token.lemma_ for token in sent2]
 
lemmatized_sentence1 = ' '.join(lemmatized_tokens1)
lemmatized_sentence2 = ' '.join(lemmatized_tokens2)

alignments, tokens1, tokens2 = align_sentences(lemmatized_sentence1, lemmatized_sentence2)

alignments = sorted(alignments, key=lambda x: x[2], reverse=True)

used_tokens1_set = set()
used_tokens2_set = set()

final_alignments = []

for alignment in alignments:
    token1, token2, similarity = alignment
    if token1 not in used_tokens1_set and token2 not in used_tokens2_set:
        final_alignments.append(alignment)
        used_tokens1_set.add(token1)
        used_tokens2_set.add(token2)

print(final_alignments)