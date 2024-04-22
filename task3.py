import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_excel("Spam Email Detection.xlsx")
df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(df['v2'].astype(str), df['v1'], test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vect, y_train)
y_pred_train = nb_classifier.predict(X_train_vect)
y_pred_test = nb_classifier.predict(X_test_vect)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)
print("Classification Report for Testing Set:")
print(classification_report(y_test, y_pred_test))