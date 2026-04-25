import pandas as pd
import nltk
import string
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("dataset.csv")

# Remove missing values
data = data.dropna()

# Load stopwords ONCE (important)
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

# Apply cleaning
data['clean_text'] = data['text'].apply(clean_text)

# Convert text to numbers
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['clean_text'])

# Labels
y_category = data['category']
y_priority = data['priority']

# Train-test split (same split for both)
from sklearn.model_selection import train_test_split

X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
    X, y_category, y_priority, test_size=0.2, random_state=42, stratify=y_category
)

# Use Logistic Regression (better than Naive Bayes)
from sklearn.linear_model import LogisticRegression

model_category = LogisticRegression(max_iter=200)
model_category.fit(X_train, y_cat_train)

model_priority = LogisticRegression(max_iter=200)
model_priority.fit(X_train, y_pri_train)

# Evaluation
from sklearn.metrics import accuracy_score

y_pred_cat = model_category.predict(X_test)
print("Category Accuracy:", accuracy_score(y_cat_test, y_pred_cat))

y_pred_pri = model_priority.predict(X_test)
print("Priority Accuracy:", accuracy_score(y_pri_test, y_pred_pri))

# Prediction function
def predict_ticket(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    category = model_category.predict(vector)[0]
    priority = model_priority.predict(vector)[0]

    print("\nTicket:", text)
    print("Predicted Category:", category)
    print("Predicted Priority:", priority)

# Test example
predict_ticket("My account is hacked and I can't login")