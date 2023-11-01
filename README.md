# IBM CAD Project - Big data analytics
## Introduction
This project exemplifies the convergence of cloud technology and big data analysis, offering a versatile and efficient solution for businesses in the digital age. By automating sentiment evaluation and employing the vast computational resources of the cloud, it enables organizations to gain a deeper understanding of their customers' sentiments and opinions at scale. This valuable insight not only informs strategic decisions but also fosters improved customer engagement and satisfaction. 
## Datasets
The dataset contains enormous number of tweets such as,feedbacks,product reviews and many user created contents.It is classified with six attributes and contains more than one million tweets. This big-data dataset was in a format of .csv file format.This dataset comprieses of positivie,negative or neutral specific emotions.The attributes contains metadata such as,date of the text,user name and other informations.The main objective of this dataset is to bringout valuable buisness insights by performing sentimental analysis to this dataset.
Link - https://drive.google.com/drive/folders/1vkNHKWCoOLhi4DaV8ag8a0xm41QJYqqF
## Data preparation and exploration
### 1.Loading Dataset:
•	It uses the `pd.read_csv()` function from the Pandas library to read a CSV file. The file is located at drive.
•	The `df.head()` function is used to display the first few rows of the dataset to get an initial look at the data.
### 2. Data Shape:
•	The code uses `df.shape` to print the dimensions of the dataset, which includes the number of rows and columns.
### 3. Data Cleaning :
•	`df.info()` provides information about the dataset, including the number of non-null entries and data types of each column.
•	`df.isnull().sum()` is used to check for null values in the dataset. The result shows that there are no null values.
### 4. Column Dropping:
•	The code removes unnecessary columns from the dataset using `df.drop()` Columns such as 'id of the tweet, date of the tweet, query, and user are dropped, leaving only sentiment and text columns.
### 5. Column Renaming:
•	The code renames the remaining columns to sentiment and text for simplicity and clarity.
### 6. Sentiment Value Counts:
•	The code uses `df['sentiment'].value_counts()` to display the counts of different sentiment labels in the sentiment column. This provides an initial understanding of the sentiment distribution in the dataset.
•	These steps are essential for setting up the dataset and understanding its basic characteristics before proceeding with sentiment analysis.
## Text Preprocessing
We need to Import Libraries and Resources such as the Natural Language Toolkit (NLTK) library and downloads the list of English stopwords. NLTK is a popular library for natural language processing tasks. we need to define `stuff_to_be_removed`, which is a list of English stopwords and punctuation symbols. These will be removed from the text data during preprocessing. Tokenization and Stemming initializes a Lancaster Stemmer, which is a stemmer used to reduce words to their root form.
The code iterates through the text in the corpus and applies various preprocessing steps to each piece of text. These steps include:

•Removing non-alphabet characters: `re.sub('[^a-zA-Z]', ' ', df['text'][i])`

•Converting text to lowercase: `text = text.lower()`

•Handling HTML tags: `re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)`

•Removing digits and non-word characters: `re.sub("(\\d|\\W)+", " ", text)`

•Tokenizing the text into a list of words: `text = text.split()`

•Stemming using the Snowball Stemmer: `text = [lem.stem(word) for word in text if not word in stuff_to_be_removed]`

•Joining the preprocessed words into a single string: `text1 = " ".join(text)`

  ```python
  data_cleaned = pd.DataFrame()
  data_cleaned["text"] = final_corpus_joined
  data_cleaned["sentiment"] = df["sentiment"].values
  ```

This code creates a new DataFrame named data_cleaned, which contains two columns: "text" and "sentiment." The "text" column contains the preprocessed and cleaned text data, while the "sentiment" column appears to be a copy of the "sentiment" column from the original DataFrame df.

## Exploratory data analysis
### 1. Creating a new DataFrame for word cloud purposes:

   ```python
   data_eda = pd.DataFrame()
   data_eda['text'] = final_corpus
   data_eda['sentiment'] = df['sentiment'].values
   ```

   This code creates a new DataFrame named `data_eda` with two columns: "text" and "sentiment." The "text" column contains the preprocessed text data from `final_corpus`, and the "sentiment" column is populated with the sentiment values from the original DataFrame `df`.

### 2. Separating text data for positive and negative sentiments:

   ```python
   positive = data_eda[data_eda['sentiment'] == 4]
   positive_list = positive['text'].tolist()
   negative = data_eda[data_eda['sentiment'] == 0]
   negative_list = negative['text'].tolist()
   ```

   This code creates two separate lists, `positive_list` and `negative_list`, containing the preprocessed text data for positive (sentiment 4) and negative (sentiment 0) sentiments, respectively.

### 3. Combining text data for positive and negative sentiments into single strings:

   ```python
   positive_all = " ".join([word for sent in positive_list for word in sent])
   negative_all = " ".join([word for sent in negative_list for word in sent])
   ```

   These lines combine all the preprocessed text data for positive and negative sentiments into single strings, `positive_all` and `negative_all`.

### 4. Creating word clouds for positive and negative sentiments:

   - For Positive Sentiment:

     ```python
     wordcloud = WordCloud(width=1000,
                           height=500,
                           background_color='magenta',
                           max_words=90).generate(positive_all)

     plt.figure(figsize=(30, 20))
     plt.imshow(wordcloud)
     plt.title("Positive")
     plt.show()
     ```

     This code uses the `WordCloud` library to create a word cloud for positive sentiment. It specifies the dimensions, background color, and the maximum number of words in the word cloud and then displays it.

   - For Negative Sentiment:

     ```python
     wordcloud = WordCloud(width=1000,
                           height=500,
                           background_color='cyan',
                           max_words=90).generate(negative_all)

     plt.figure(figsize=(30, 20))
     plt.imshow(wordcloud)
     plt.title("Negative")
     plt.show()
     ```

     Similarly, this code generates a word cloud for negative sentiment with specified dimensions, background color, and a maximum number of words and displays it.
These word clouds visualize the most common words in the text data for positive and negative sentiments, providing an overview of the most frequent words associated with each sentiment. The word clouds are displayed using Matplotlib's `imshow` function.

## Model Building
### 1. Data Preparation:

   - `X` and `y` are defined to represent the feature (text) and target (sentiment) variables, respectively, using the cleaned data from the `data_cleaned` DataFrame.

   - The `TfidfVectorizer` from scikit-learn is used to convert the text data into a numerical format. The `fit_transform` method is applied to the feature data `X`, resulting in a term frequency-inverse document frequency (TF-IDF) representation of the text data stored in `Xt`.

   - The dataset is split into training and testing sets using `train_test_split`. The features (`Xt`) and target (`y`) are split into `X_train`, `X_test`, `y_train`, and `y_test`. The testing set size is set to 20% of the data, and a random seed (`random_state`) is specified for reproducibility.

### 2. Logistic Regression Model:

   - A logistic regression model is initialized using the `LogisticRegression` class from scikit-learn:

     ```python
     model = LogisticRegression()
     ```

   - The logistic regression model is trained using the training data:

     ```python
     model.fit(X_train, y_train)
     ```

### 3. Making Predictions:

   - The trained logistic regression model is used to make predictions on the testing data:

     ```python
     y_pred = model.predict(X_test)
     ```

   The variable `y_pred` now contains the predicted sentiment labels for the test data.

This code builds a sentiment analysis model using TF-IDF vectorization and logistic regression classification. The model is trained on the training data and then used to predict sentiment labels for the test data. You can further evaluate the model's performance using appropriate evaluation metrics (e.g., accuracy, precision, recall, F1-score) to assess its predictive capabilities.

## Model Evaluation
### 1. Classification Report:

   ```python
   from sklearn.metrics import classification_report
   print('Classification Report:\n', classification_report(y_test, y_pred))
   ```

   This code computes a classification report for the model's predictions. The `classification_report` function from scikit-learn is used to generate a report that includes metrics such as precision, recall, F1-score, and support for each class in the target variable. The report is printed to the console.

### 2. Confusion Matrix:

   ```python
   from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
   t1 = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
   t1.plot()
   ```

   This code calculates the confusion matrix for the model's predictions. The confusion matrix displays the true positive, true negative, false positive, and false negative values, which are used to evaluate the model's performance. The `ConfusionMatrixDisplay` class is used to visualize the confusion matrix, and the `plot` method displays the matrix as a graphical representation.

## Conclusion
Overall, the code provides a solid foundation for sentiment analysis, from data preprocessing to model building and evaluation. The process can be further refined and expanded, and additional machine learning models and techniques can be explored to improve sentiment classification accuracy. This code serves as a valuable starting point for sentiment analysis tasks on textual data and it can be adapted and extended to suit the specific needs of different applications and industries.

