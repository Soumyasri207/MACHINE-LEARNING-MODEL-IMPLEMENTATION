# Spam Detection with Visual Output using Machine Learning

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Step 2: Create Dataset
data = {
    "message": [
        "Congratulations! You won a free lottery ticket",
        "Hello, how are you doing today?",
        "Claim your free prize now",
        "Meeting at 5 PM today",
        "Win cash rewards instantly",
        "Can we complete the project tomorrow?",
        "Exclusive offer just for you",
        "Please send the assignment file",
        "Get free recharge now",
        "Let's go for lunch"
    ],

    "label": [
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham"
    ]
}

# Step 3: Convert into DataFrame
df = pd.DataFrame(data)

# Step 4: Split Data
x = df["message"]
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Step 5: Convert Text into Numbers
vectorizer = CountVectorizer()

x_train_features = vectorizer.fit_transform(x_train)
x_test_features = vectorizer.transform(x_test)

# Step 6: Train Model
model = MultinomialNB()
model.fit(x_train_features, y_train)

# Step 7: Predictions
y_pred = model.predict(x_test_features)

# Step 8: Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Step 9: Visualize Spam vs Ham Count
plt.figure(figsize=(6,4))

df['label'].value_counts().plot(kind='bar')

plt.title("Spam vs Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Count")

plt.show()

# Step 10: Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()

plt.title("Confusion Matrix")
plt.show()

# Step 11: Test Custom Message
sample_message = ["You have won free money"]

sample_data = vectorizer.transform(sample_message)

prediction = model.predict(sample_data)

print("\nMessage:", sample_message[0])
print("Prediction:", prediction[0])
