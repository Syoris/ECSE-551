from sklearn.feature_extraction.text import CountVectorizer


# Custom preprocessor function to keep strings with '$'
def custom_preprocessor(text):
    # Split the text by spaces and keep words containing '$'
    words = [word for word in text.split() if '$' in word]
    return ' '.join(words)


def custom_tokenizer(text):
    return text.split()


# Sample text data
text_data = ['This is a $100 example', 'Another example without $ sign', 'Yet another $200 example']

# Create an instance of CountVectorizer with the custom preprocessor
vectorizer = CountVectorizer(tokenizer=custom_tokenizer, lowercase=False)

# Fit and transform the text data
vectorized_data = vectorizer.fit_transform(text_data)

# Get feature names
feature_names = vectorizer.get_feature_names_out()
print(feature_names)

# Convert vectorized data to a DataFrame for visualization
import pandas as pd

df = pd.DataFrame(vectorized_data.toarray(), columns=feature_names)
print(df)
