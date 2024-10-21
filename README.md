# Semantic Textual Similarity (STS) Project

## Project Overview

This project aims to develop a system that automatically measures the semantic similarity between two sentences. The system assigns a score ranging from 0 to 5, where a score close to 5 indicates that the sentences have similar meanings, while a score near 0 suggests that they are semantically different. The primary focus of this project is on monolingual (English-English) sentence pairs.

## Problem Statement

The goal of this project is to create a model that can predict the semantic similarity between two sentences. For example, given two sentences, the model will assign a continuous similarity score based on their meaning. A score of 5 means the sentences have very similar meanings, while a score of 0 indicates completely different meanings. This project leverages machine learning techniques to develop such a model, which is particularly useful in tasks like paraphrase detection, text summarization, and question answering.

## Datasets

### 1. **SICK Dataset (Sentences Involving Compositional Knowledge)**  
The SICK dataset is designed to study compositional distributional semantics and contains sentence pairs annotated for both relatedness and entailment.  
- **Similarity Score**: Originally on a scale of 1 to 5, which we normalized to 0-5.  
- **Link**: [SICK Dataset on HuggingFace](https://huggingface.co/datasets/sick)

### 2. **STS Benchmark Dataset**  
The STS Benchmark dataset includes 8,628 sentence pairs sourced from three domains: news, image captions, and forum discussions. It was used in Semantic Textual Similarity tasks during SemEval competitions from 2012 to 2017.  
- **Similarity Score**: Human-annotated scores for reliable model training and evaluation.  
- **Link**: [STS Benchmark on HuggingFace](https://huggingface.co/datasets/mteb/stsbenchmark-sts)

## Methods

The project consists of three core methods, which are combined in an ensemble for final predictions:

### 1. **Statistical Approach** (Method 1)  
This approach relies on traditional statistical methods to assess textual similarity.

### 2. **Neural Network-based Approach** (Method 2)  
This approach implements deep learning models like CNN, RNN, and LSTM to extract and model semantic features from the text.

### 3. **BERT Fine-tuning** (Method 3)  
This method uses BERT, a pre-trained transformer model, and fine-tunes it for the STS task to enhance semantic understanding.

The ensemble approach combines these methods for improved performance across datasets.

## Model Performance

### STS Benchmark Dataset:
- **LSTM**: Pearson's correlation coefficient of 0.707
- **BERT**: Pearson's correlation coefficient of 0.741
- **CNN**: Pearson's correlation coefficient of 0.628
- **RNN**: Pearson's correlation coefficient of 0.619

### SICK Dataset:
- **CNN**: Pearson's correlation coefficient of 0.8059
- **LSTM**: Pearson's correlation coefficient of 0.799
- **BERT**: Pearson's correlation coefficient of 0.736
- **RNN**: Performed poorly on this dataset

### Ensemble Model:
- **STS Benchmark**: Pearson's correlation coefficient of 0.760
- **SICK Dataset**: Pearson's correlation coefficient of 0.8579

The ensemble method, which combines RNN, CNN, LSTM, and BERT, significantly improves performance across both datasets, effectively capturing nuanced semantic relationships.

## Analysis and Results

### STS Benchmark Dataset:
- **Best Model**: BERT
- **Best Pearson’s Correlation**: 0.741 (BERT)

### SICK Dataset:
- **Best Model**: CNN
- **Best Pearson’s Correlation**: 0.8059 (CNN)

The ensemble method outperforms individual models on both datasets, demonstrating the effectiveness of combining multiple architectures for better semantic understanding.

## Conclusion

This project demonstrates the application of various machine learning models to the problem of Semantic Textual Similarity (STS). Through experimentation with different architectures and the use of an ensemble approach, the project highlights the strengths and limitations of each method in capturing semantic similarities between sentences. The ensemble method, in particular, achieves strong results across datasets, showcasing its potential for real-world natural language understanding tasks.
 
