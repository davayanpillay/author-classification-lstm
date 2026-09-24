# Author Classification Using Bidirectional LSTM

## Overview

This project develops a five-class Natural Language Processing model to predict the author of a sentence based on writing style.

The model classifies text from five nineteenth-century authors:

- Charles Dickens
- Herman Melville
- Jane Austen
- Louisa May Alcott
- Mark Twain

PySpark was used for data loading, cleaning and exploratory data analysis, while TensorFlow/Keras was used to build and train the recurrent neural network models.

## Technologies

- Python
- PySpark
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Natural Language Processing
- LSTM
- Bidirectional LSTM
- Jupyter Notebook

## Dataset

The project uses a public sentence-level dataset containing text from five nineteenth-century novelists.

The original dataset contained more than 115,000 records.

After cleaning and duplicate removal, a balanced modelling dataset of 50,000 sentences was created using 10,000 sentences per author.

Dataset source:

https://huggingface.co/datasets/Mosab-Rezaei/19th-century-novelists/tree/main

The raw dataset is not included in this repository because of its file size.

## Methodology

The workflow included:

1. Data loading using PySpark
2. Missing-value and duplicate analysis
3. Exploratory data analysis
4. Author class distribution analysis
5. Sentence-length analysis
6. Feature and target selection
7. Class balancing
8. Text cleaning and normalisation
9. Label encoding
10. Stratified train-test split
11. Tokenisation
12. Sequence padding and truncation
13. Baseline LSTM modelling
14. Bidirectional LSTM model improvement
15. Final evaluation on an untouched test set
16. Development of a simple author-prediction bot

## Model Development

### Baseline LSTM

The baseline model used:

- 5,000-word vocabulary
- 64-dimensional embedding layer
- 64-unit LSTM layer
- 5-class softmax output
- Adam optimiser
- Sparse categorical cross-entropy
- Early stopping

Best validation accuracy:

**67.58%**

Best validation loss:

**0.9125**

### Improved Bidirectional LSTM

The improved model introduced:

- Bidirectional LSTM
- Dropout regularisation
- Additional dense layer

Best validation accuracy:

**70.00%**

Best validation loss:

**0.8177**

## Final Test Results

The improved model was evaluated on an untouched test set containing 10,000 sentences.

- Test Accuracy: **68.66%**
- Test Loss: **0.8132**
- Macro F1-score: **0.6897**
- Weighted F1-score: **0.6897**

### Per-Author F1 Scores

- Jane Austen: **0.751**
- Louisa May Alcott: **0.745**
- Herman Melville: **0.671**
- Charles Dickens: **0.653**
- Mark Twain: **0.627**

The model performed particularly well when identifying Jane Austen and Louisa May Alcott, while some overlap was observed between authors such as Herman Melville and Mark Twain.

## Book Bot

A simple prediction function was developed to allow new sentences to be entered and classified as one of the five authors.

The prediction pipeline applies the same text preprocessing, tokenisation and sequence padding used during model training.

## Limitations

The model operates as a closed-set classifier and can only predict one of the five authors included in the training data.

Because the sentences were extracted from literary works, the model may also learn recurring names, topics and book-specific vocabulary in addition to writing style.

Performance should therefore not be generalised to modern authors or completely different literary domains.

## Repository Contents

- `author_classification_lstm.ipynb` — complete modelling workflow
- `author_classification_lstm_planning.pdf` — project planning and methodology
- `author_classification_lstm_report.pdf` — final report and evaluation

## Author

Davayan Pillay  
BSc Information Technology (Data Science)  
Postgraduate Diploma in Data Analytics
