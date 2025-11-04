# Syntecxhub_Spam_Detection

### 🎯 **Objective**
To build a machine learning model that detects whether a given text message is **Spam** (unwanted) or **Ham** (legitimate).  
Then integrate it into an **interactive Command Line Interface (CLI)** that allows real-time predictions.

### 📊 **Dataset**
- **Name:** SMS Spam Collection Dataset  
- **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
- **Samples:** ~5,500 text messages  
- **Classes:**  
  - **Ham (0):** Normal messages  
  - **Spam (1):** Unwanted / promotional messages
 
  - ### ⚙️ **Technologies Used**
- **Language:** Python  
- **Libraries:**  
  - pandas, numpy — data handling  
  - nltk — text preprocessing & stopwords removal  
  - scikit-learn — TF-IDF vectorization, Naive Bayes model, pipeline  
  - joblib — model saving/loading  

- **Environment:** Google Colab  

### 🪜 **Project Workflow**

1. **Data Loading**
   - Loaded labeled spam/ham dataset from Kaggle.
   - Extracted key columns (label, message).

2. **Data Cleaning**
   - Converted all text to lowercase.
   - Removed numbers, punctuation, and extra spaces.
   - Used **NLTK stopwords** to remove unimportant words.

3. **Text Vectorization**
   - Used **TF-IDF Vectorizer** to convert cleaned text into numeric form.
   - Limited vocabulary to top 3,000 words for efficiency.
  
4. **Model Training**
   - Trained **Multinomial Naive Bayes** model — best for text classification.
   - Split data (80% train / 20% test).

5. **Evaluation**
   - Achieved ~98% accuracy.
   - Analyzed confusion matrix and misclassified samples.
  
6. **Pipeline Creation**
   - Combined **TF-IDF Vectorizer + Naive Bayes** into one pipeline.
   - Saved as `spam_detection_pipeline.pkl`.

7. **Interactive CLI**
   - Built a simple console-based tool for live message predictions.  
   - User can type messages and get instant results.

Metric	Score
Accuracy	0.98
Precision	0.97
Recall	0.88
F1-score	0.92

✅ Conclusion:
The Naive Bayes classifier, combined with TF-IDF, achieved excellent performance for short-text spam detection.
