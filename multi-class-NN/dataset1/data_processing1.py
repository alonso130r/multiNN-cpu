import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load each dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
validation_df = pd.read_csv('validation.csv')


# split into x, y
X_train = train_df['sentence']
y_train = train_df['idx']
X_test = test_df['sentence']
y_test = test_df['idx']
X_validation = validation_df['sentence']
y_validation = validation_df['idx']

# initialize vectorizer
vectorizer = TfidfVectorizer(max_features=1000)

# fit vectorizer on train data + transform
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()

# transform test/val
X_test_tfidf = vectorizer.transform(X_test).toarray()
X_validation_tfidf = vectorizer.transform(X_validation).toarray()

# convert labels to numpy arrays
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)
y_validation_np = np.array(y_validation)

# export to csv (train)
train_df = pd.DataFrame(X_train_tfidf)
train_df['label'] = y_train_np
train_df.to_csv('train_pre.csv', index=False)

# export to csv (test)
test_df = pd.DataFrame(X_test_tfidf)
test_df['label'] = y_test_np
test_df.to_csv('test_pre.csv', index=False)

# export to csv (val)
validation_df = pd.DataFrame(X_validation_tfidf)
validation_df['label'] = y_validation_np
validation_df.to_csv('val_pre.csv', index=False)
