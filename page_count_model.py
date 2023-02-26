import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


class BookModel:
    def __init__(self, file_path):
        # Load the dataset
        df = pd.read_excel(file_path)

        # Clean the data
        df.dropna(inplace=True)
        df['finish'] = df['finish'].map({'yes': 1, 'no': 0})

        # Engineer features
        df['percentage_complete'] = df['page_count'] / df['page_count'].max()
        df['pages_remaining'] = df['page_count'].max() - df['page_count']

        self.X = df[['page_count', 'percentage_complete', 'pages_remaining']]
        self.y = df['finish']

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Create an instance of the LogisticRegression estimator
        self.model = LogisticRegression()

        # Fit the model on the training data
        self.model.fit(self.X_train, self.y_train)

    def predict(self, page_count):
        percentage_complete = page_count / self.X['page_count'].max()
        pages_remaining = self.X['page_count'].max() - page_count
        X_new = [[page_count, percentage_complete, pages_remaining]]

        prediction = self.model.predict(X_new)
        if prediction[0]:
            return 'You will be able to finish the book!'
        else:
            return 'You will not be able to finish the book'

    def plot(self):
        # Make predictions on the testing data
        probs = self.model.predict_proba(self.X_test)

        # Get the probabilities for the "yes" class
        yes_probs = probs[:, 1]

        # Plot page count vs. probability of the book being read
        plt.scatter(self.X_test['page_count'], yes_probs)
        plt.xlabel('Page count')
        plt.ylabel('Probability of book being read')
        plt.title('Page count vs. probability of book being read')
        plt.show()


model = BookModel('ml-assessment-data.xlsx')
print(model.predict(1200))
model.plot()
