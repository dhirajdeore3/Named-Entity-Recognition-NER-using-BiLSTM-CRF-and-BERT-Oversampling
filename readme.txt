									

---

# Named Entity Recognition (NER) using BiLSTM-CRF and BERT Oversampling

This repository contains code for building a Named Entity Recognition (NER) model using a combination of BiLSTM-CRF and BERT oversampling techniques. The goal is to identify and classify named entities in text data, such as people, organizations, locations, etc.

## Introduction
Named Entity Recognition (NER) is a task in natural language processing (NLP) that aims to identify and classify named entities in text. Named entities can be people, organizations, locations, or other entities. NER is a challenging task because the boundaries of named entities are often ambiguous and can be difficult to identify.

In this notebook, we will build a NER model using a combination of BiLSTM-CRF and BERT oversampling.

- BiLSTM-CRF is a well-known model for NER that has been shown to achieve state-of-the-art results.
- BERT is a pre-trained language model that has been shown to be effective for a variety of NLP tasks, including NER.

We will oversample the minority classes in our dataset to address the class imbalance problem. Class imbalance is a common problem in NER datasets, and it can make it difficult for the model to learn to identify the minority classes. Oversampling the minority classes will help the model to learn to identify these classes more accurately.

We will evaluate our model on the Given dataset. We will use the F1 score to measure the performance of our model.

## Data Import and Analysis
The data import and analysis section preprocesses the dataset, handles class imbalances, and performs exploratory data analysis (EDA) to understand the distribution of named entities in the dataset. It also visualizes the class distribution using histograms and checks for null values in the dataset.

## Model Building: LSTM and CRF
### Data Preprocessing
Data preprocessing involves tokenization, padding, and encoding of text data and labels. It prepares the data for model training by converting text tokens to numerical representations using word embeddings and one-hot encoding for labels.

### Model Building
The model building section creates a BiLSTM-CRF model for NER using TensorFlow/Keras. It defines the architecture of the model, including embedding layers, bidirectional LSTM layers, and a CRF layer for sequence labeling. It also sets hyperparameters such as batch size, epochs, and learning rate.

### Evaluation
The evaluation section measures the performance of the BiLSTM-CRF model on the test dataset using metrics such as accuracy, F1 score, and confusion matrix. It visualizes the model's performance and discusses the results.

### Conclusion
The conclusion summarizes the findings from the BiLSTM-CRF model, highlighting its strengths and areas for improvement. It suggests possible future enhancements or experiments to improve NER model performance.

## OverSampling and BERT
The Oversampling and BERT section focuses on addressing class imbalances using oversampling techniques and incorporating BERT embeddings for NER. It includes steps for preprocessing, tokenization using BERT's tokenizer, model building with BERT, evaluation, and conclusion.

### Oversampling and Preprocessing
Oversampling is performed to balance class distribution in the dataset, followed by preprocessing steps such as label encoding and tokenization using BERT's tokenizer. It prepares the data for BERT model training by converting text tokens to input IDs and attention masks.

### Bert Modeling
The Bert Modeling section creates a BERT-based model for NER using TensorFlow/Keras and the Hugging Face Transformers library. It defines the model architecture, including BERT embeddings, dropout layers, and dense layers for classification. It also sets hyperparameters and callbacks for model training.

### Evaluation
The evaluation section assesses the performance of the BERT-based model on the test dataset using metrics like accuracy, precision, recall, and F1 score. It visualizes the model's performance using a confusion matrix and discusses the results compared to the BiLSTM-CRF model.

### Conclusion
The conclusion summarizes the findings from the BERT-based model, highlighting improvements in accuracy and category-wise performance. It discusses the impact of oversampling and BERT embeddings on NER model performance and suggests potential areas for further research.

---

