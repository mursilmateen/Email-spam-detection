# ==============================
# 📌 EMAIL SPAM DETECTION SYSTEM
# ==============================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
import nltk

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns

# ==============================
# 1. Load Dataset
# ==============================
print("📂 Loading dataset...")
spam = pd.read_csv(r"C:\Users\mursa\OneDrive\Desktop\spam.csv", encoding='ISO-8859-1')

# Keep only relevant columns
spam = spam[['v1', 'v2']]
spam.columns = ['label', 'message']

print("\n✅ Dataset Loaded Successfully!")
print(spam.head())

# ==============================
# 2. Data Preprocessing
# ==============================
print("\n🔄 Preprocessing messages...")

ps = PorterStemmer()
corpus = []

for i in range(0, len(spam)):
    review = re.sub('[^a-zA-Z]', ' ', spam['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Bag of Words Model
cv = CountVectorizer(max_features=4000)
X = cv.fit_transform(corpus).toarray()

# Convert labels
Y = pd.get_dummies(spam['label'])
Y = Y.iloc[:, 1].values  # spam = 1, ham = 0

# ==============================
# 3. Train-Test Split
# ==============================
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# ==============================
# 4. Train Models
# ==============================
print("\n🤖 Training models...")

# Random Forest
model1 = RandomForestClassifier()
model1.fit(X_train, Y_train)

# Decision Tree
model2 = DecisionTreeClassifier()
model2.fit(X_train, Y_train)

# Multinomial Naïve Bayes
model3 = MultinomialNB()
model3.fit(X_train, Y_train)

# ==============================
# 5. Predictions
# ==============================
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)

# ==============================
# 6. Evaluation
# ==============================
print("\n📊 Model Performance:\n")

print("Random Forest Classifier")
print(confusion_matrix(Y_test, pred1))
print("Accuracy:", accuracy_score(Y_test, pred1))
print(classification_report(Y_test, pred1))
print("--------------------------------")

print("Decision Tree Classifier")
print(confusion_matrix(Y_test, pred2))
print("Accuracy:", accuracy_score(Y_test, pred2))
print(classification_report(Y_test, pred2))
print("--------------------------------")

print("Multinomial Naïve Bayes")
print(confusion_matrix(Y_test, pred3))
print("Accuracy:", accuracy_score(Y_test, pred3))
print(classification_report(Y_test, pred3))
print("--------------------------------")

# Confusion Matrix Heatmap for Naïve Bayes
cm = confusion_matrix(Y_test, pred3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naïve Bayes")
plt.show()

# ==============================
# 7. Save Models
# ==============================
pickle.dump(model1, open("RFC.pkl", "wb"))
pickle.dump(model2, open("DTC.pkl", "wb"))
pickle.dump(model3, open("MNB.pkl", "wb"))
print("\n💾 Models saved successfully as RFC.pkl, DTC.pkl, MNB.pkl")

# ==============================
# 8. Prediction Function
# ==============================
def predict_email(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    vector = cv.transform([review]).toarray()
    prediction = model3.predict(vector)
    return "🚨 Spam Email" if prediction[0] == 1 else "✅ Not Spam"

# ==============================
# 9. Test with Custom Inputs
# ==============================
print("\n🔍 Testing on Custom Emails:\n")
print("Test 1:", predict_email("FREE entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 12345"))
print("Test 2:", predict_email("URGENT! You have won $1000 cash."))
print("Test 3:", predict_email("Hello Mursal, how are you doing?."))
