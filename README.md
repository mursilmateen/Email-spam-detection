📧 Email Spam Detection

Spam emails are one of the most common issues in digital communication. The purpose of this project is to create a machine learning-based system that can automatically classify emails or text messages as spam or not spam (ham).

This project makes use of Python and Natural Language Processing (NLP) techniques to clean and process text data, followed by training machine learning models for classification.

🔹 What I Did in This Project

Downloaded the dataset from Kaggle: SMS Spam Collection Dataset

Preprocessed the text data (cleaning, lowercasing, removing stopwords, stemming)

Converted text into numerical features using CountVectorizer (Bag of Words)

Trained three machine learning models:

Random Forest Classifier

Decision Tree Classifier

Multinomial Naïve Bayes

Evaluated model performance using confusion matrix, accuracy score, and classification reports

Saved the trained models with pickle for future use

🔹 Tools and Libraries Used

Python

Pandas, NumPy

Scikit-learn

NLTK

Matplotlib, Seaborn

🔹 How It Works

User inputs an email text.

Text is preprocessed using NLP methods.

Processed text is converted into numerical vectors.

The trained model predicts whether the email is spam or not.

The output is shown as Spam or Not Spam.

🔹 Example

“You have won a free iPhone!” → Spam
“Hello Mursal, let’s meet tomorrow.” → Not Spam

🔹 Results

Naïve Bayes performed the best among the three models.

The system was able to correctly identify most spam messages containing common spam keywords such as win, free, claim, prize.

🔹 Future Improvements

Use TF-IDF Vectorizer for more accurate text representation.

Test with deep learning models for higher accuracy.

Deploy as a Flask or Django web application for real-time usage.

🔹 My Note

This project was developed as part of my learning in Data Science and Machine Learning. Through this, I understood how text preprocessing, NLP, and supervised learning models can be combined to build a working spam detection system.

✍️ Created by Mursal Mateen | Semester Project– Spam Detection Project
