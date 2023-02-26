class BookModel:
    def __init__(self, df_train):
        self.df_train = df_train
        self.category_counts = df_train['book_category'].value_counts(normalize=True)
        self.category_success = df_train.groupby('book_category')['finish'].mean()
        self.finish_status = df['finish'].map({'yes': 1, 'no': 0})
        self.category_stats = pd.concat([self.category_counts, self.category_success], axis=1)
        self.category_stats.columns = ['frequency', 'success_rate']
    
    def predict_success_rate(self, book_category):
        if book_category in self.category_stats.index:
            success_rate = self.category_stats.loc[book_category, 'success_rate']
            return success_rate
        else:
            return None
    
    def predict_finish_status(self, page_count, book_category):
        if book_category in self.category_stats.index:
            success_rate = self.category_stats.loc[book_category, 'success_rate']
            if success_rate >= 0.5:
                return 'yes'
            else:
                return 'no'
        else:
            return 'unknown'


df_train = pd.read_excel('ml-assessment-data.xlsx')
df_train['finish'] = df['finish'].map({'yes': 1, 'no': 0})
book_model = BookModel(df_train)
book_category = 'Fantasy' # replace with user input
success_rate = book_model.predict_success_rate(book_category)
print(f"The predicted success rate for {book_category} is {success_rate:.2f}")

page_count = 300 # replace with user input
finish_status = book_model.predict_finish_status(page_count, book_category)
print(f"Will you finish the {book_category} book? {finish_status}")

    
