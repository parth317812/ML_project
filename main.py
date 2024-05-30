import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

raw_mail_data = pd.read_csv('mail_data.csv')

# replace the null values wiht a null string

mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# label spam mail as 0, ham mail as 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

x = mail_data['Message']
y = mail_data['Category']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# trainsform the test data to feature vectors that can be used as input to the logistic regression model
print("x_train: ", x_train)
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

# convert targets to integers

y_train = y_train.astype('int')
y_test = y_test.astype('int')

regressor = LogisticRegression()

regressor.fit(x_train_features, y_train)

pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

print(x_train_features)
prediction_training = model.predict(x_train_features)

accuracy_on_Training_data = accuracy_score(y_train, prediction_training)

prediction_test = model.predict(x_test_features)

accuracy_on_Test_data = accuracy_score(y_test, prediction_test)
print(x_test_features[501])
prediction = model.predict(x_test_features[501])

if prediction[0] == 1:
    print("It's not a spam mail")
else:
    print("It's a spam mail")
