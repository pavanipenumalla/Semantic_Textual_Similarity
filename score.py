from alignments import get_alignments
import nltk

def get_similarity_score(sent1, sent2):
    alignments,c1,c2 = get_alignments(sent1, sent2)
    alignment_score_sum = 0
    
    for alignment in alignments:
        t1,t2,similarity = alignment
        t1 = t1.split('_')
        t2 = t2.split('_')
        t1 = [word for word in t1 if word.lower() not in nltk.corpus.stopwords.words('english')]
        t2 = [word for word in t2 if word.lower() not in nltk.corpus.stopwords.words('english')]
        factor = len(t1) + len(t2)
        alignment_score_sum += factor*similarity
    
    return (alignment_score_sum/(c1+c2))*5