from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample training messages
messages = [
    "Win a free prize now",
    "Congratulations you won a lottery",
    "Claim your free money",
    "You have won a cash prize",
    "Get free rewards now",
    "Hello how are you",
    "Please submit your assignment",
    "Can we meet tomorrow",
    "Your class starts at 10 AM",
    "Please call me when you are free"
]

# Labels for the messages
labels = [
    "Spam",
    "Spam",
    "Spam",
    "Spam",
    "Spam",
    "Not Spam",
    "Not Spam",
    "Not Spam",
    "Not Spam",
    "Not Spam"
]

# Convert text into numbers
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(messages)

# Create and train the model
model = MultinomialNB()

model.fit(X, labels)

# Get message from user
message = input("Enter a message: ")

# Convert user message into numbers
message_vector = vectorizer.transform([message])

# Predict the message
prediction = model.predict(message_vector)

# Display result
print("\nResult:", prediction[0])