# Quora Question Pair Similarity Project

## Overview

Quora is a popular platform where users ask questions and receive answers from the community. One of the challenges is identifying whether two different questions are semantically similar (duplicates) or not. This project aims to classify pairs of questions as duplicates or non-duplicates, improving the content relevance and minimizing redundancy on Quora.

## Objective

The primary objective of this project is to classify pairs of questions to determine whether they are duplicates, thereby enhancing content quality and user experience on Quora.

## Approach and Methodology

The project employs two different methods to tackle the problem:

1. **GloVe Embeddings and LSTM Network**
2. **TF-IDF Vectorization and Machine Learning Models**

### 1. GloVe Embeddings and LSTM Network

### Data Preprocessing

- **Lowercasing**: All text data is converted to lowercase to maintain consistency.
- **Number Conversion**: Numbers are converted to words using the `num2words` library.
- **Stemming**: Applied Snowball Stemming to reduce words to their base forms.
- **Stop Words Removal**: Removed common stop words to reduce noise in the data.

### Sequence Preparation

- **Tokenization**: The text data is tokenized, and sequences are prepared for both questions in each pair.
- **Padding**: The sequences are padded to a fixed length to ensure uniformity.

### Embedding Layer

- **GloVe Embeddings**: Pre-trained GloVe embeddings (200d) are used to initialize the embedding layer. The embeddings capture semantic and syntactic meanings of words.

### Neural Network Architecture

- **Bidirectional LSTM**: Two layers of Bidirectional LSTMs are used to capture dependencies in both directions.
- **Attention Mechanism**: Applied attention mechanism to focus on important parts of the questions.
- **Concatenation and Flattening**: The outputs of the attention layers are concatenated and flattened.
- **Dense Layer**: A dense layer with a sigmoid activation function is used for the final binary classification.

### Training and Evaluation

- **Early Stopping and Model Checkpointing**: Implemented to prevent overfitting and save the best model.
- **Results**: The model achieved significant improvement during the training epochs.

`Accuracy: 82.48%`
`Precision: 83%`
`Recall: 82%`
`F1 Score: 83%`

### 2. TF-IDF Vectorization and Machine Learning Models

### Data Preprocessing

- **Cleaning Text**: Converted text to lowercase, removed special characters, short words, and stop words, and applied lemmatization.
- **TF-IDF Vectorization**: Used TF-IDF vectorizer to transform the questions into vector representations.

### Feature Extraction

- **Cosine Similarity**: Calculated cosine similarity between TF-IDF vectors of the question pairs.

### Machine Learning Models

- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier**

### Training and Evaluation

- **Train-Test Split**: Split the data into training and validation sets.
- **Model Training**: Trained each model and evaluated their performance using accuracy and classification report.

 **Logistic Regression:**

`Accuracy: 64.86%`
`Precision: 63%`
`Recall: 65%`
`F1 Score: 64%`

**Random Forest:**

`Accuracy: 65.05%`
`Precision: 66%`
`Recall: 65%`
`F1 Score: 65%`

**XGBoost:**

`Accuracy: 64.92%`
`Precision: 63%`
`Recall: 65%`
`F1 Score: 63%`

## Results and Performance Comparison

### GloVe Embeddings and LSTM Network

- **Training Accuracy**: Improved accuracy and reduced loss over epochs with accuracy of 87.73%.
- **Attention Mechanism**: Enhanced the model's focus on important parts of the questions, contributing to better performance.

### TF-IDF Vectorization and Machine Learning Models

- **Logistic Regression**: Achieved an accuracy of 64.86%.
- **Random Forest Classifier**: Slightly better with an accuracy of 65.05%.
- **XGBoost Classifier**: Achieved an accuracy of 64.92%.

### Summary of Findings

- **Neural Network Approach**: The GloVe embeddings and LSTM-based model demonstrated a stronger ability to capture semantic similarity, leading to better performance in identifying duplicate questions.
- **Machine Learning Approach**: While simpler and faster to train, the TF-IDF-based models showed lower performance compared to the neural network model.

## Conclusion

The project successfully implemented two approaches to classify duplicate question pairs on Quora. The GloVe embeddings with an LSTM network showed superior performance compared to traditional machine learning models using TF-IDF vectorization. This highlights the importance of capturing semantic meaning through advanced embedding techniques and deep learning architectures.

## Future Scope

- **Enhanced Model Performance**: Continuously refining the model architecture and training process can lead to improved performance.
- **Handling Semantic Equivalence**: Incorporating more advanced contextual embeddings could enhance the model's ability to understand the underlying meaning of questions.
- **Handling Short and Noisy Questions**: Developing strategies to handle such questions can improve the model's robustness.
- **Online Learning and Real-Time Detection**: Adapting the model to an online learning framework would enable real-time duplicate detection.
- **Multilingual Duplicate Detection**: Expanding the model to handle multiple languages would extend its usefulness to a broader user base.

In conclusion, the project demonstrates the power of machine learning and deep learning in solving real-world challenges and improving user experience on platforms like Quora.
