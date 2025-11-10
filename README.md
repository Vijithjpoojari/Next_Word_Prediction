# 🧠 Next Word Prediction using Deep Learning in NLP

## 📘 Project Overview
This project focuses on building a **Next Word Prediction** model using **Deep Learning** and **Natural Language Processing (NLP)** techniques.  
The model predicts the most probable next word in a given sentence fragment by learning from a custom dataset of English sentences.  
It demonstrates how recurrent neural networks (specifically LSTMs) can understand and generate natural language.

---

## 🧭 Objective
To develop a **deep learning-based NLP model** capable of predicting the next word in a given sequence using **LSTM (Long Short-Term Memory)** networks.

---

## 🧰 Tools and Technologies Used
- **Programming Language:** Python  
- **Deep Learning Library:** TensorFlow / Keras  
- **Data Preprocessing:** Tokenizer, Padding Sequences  
- **Model Architecture:** LSTM (Long Short-Term Memory)  
- **Corpus Storage:** `corpus.txt` (custom dataset stored on Google Drive)  
- **Development Platform:** Google Colab  

---

## 📚 Dataset Used
- **Type:** Custom text corpus  
- **Size:** 100 manually written English sentences  
- **Content:** General daily life expressions and conversations  
- **File:** `corpus.txt`

This dataset is used to train the model to learn common sentence patterns and predict the next word accordingly.

---

## ⚙️ Methodology

### 1. Data Collection
- Created a custom text file (`corpus.txt`) with 100 sentences.  
- Uploaded the dataset to Google Drive and accessed it through Google Colab.

### 2. Data Preprocessing
- **Tokenization:** Convert text into integer sequences.  
- **N-gram Generation:** Create input-output pairs for each possible sequence.  
- **Padding:** Apply zero-padding to ensure uniform input lengths.  
- **One-hot Encoding:** Encode target labels for classification.

### 3. Model Building
The model includes:
- **Embedding Layer:** Converts words into dense vector representations.  
- **LSTM Layer:** Learns sequential dependencies in language.  
- **Dense Output Layer:** Predicts the next word from the vocabulary.

### 4. Model Training
- **Loss Function:** Categorical Cross-Entropy  
- **Optimizer:** Adam  
- **Epochs:** 300  

### 5. Prediction
- User provides an input sentence fragment.  
- The model predicts the most probable **next word** based on learned patterns.

---

## 💻 Sample Code (Snippet)
```python
# Example of prediction function
def predict_next_word(model, tokenizer, text_seq, max_seq_len):
    token_list = tokenizer.texts_to_sequences([text_seq])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None

# Example Input/Output
Input: "I love"
Output: "reading"
