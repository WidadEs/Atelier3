# Atelier3
## Part1 : Language Modeling / Regression
### Text Cleaning and Preprocessing
#### The clean_text function serves a pivotal role in text preprocessing by normalizing, cleaning, and standardizing raw text data. It begins by converting all text to lowercase to ensure uniformity, then removes punctuation marks to simplify the text and reduce noise. Subsequently, it eliminates numerical digits from the text, which can be irrelevant or distracting in certain analyses. This process of standardization and cleaning ensures that the output text is consistent and ready for downstream tasks such as natural language processing (NLP), sentiment analysis, or machine learning modeling. By applying this function to a DataFrame column using the apply method, it facilitates efficient batch processing of text data, making it an essential step in text data preparation pipelines.
### Tokenization, stop words removal, and lemmatization
#### The process outlined involves several steps in text preprocessing. It starts by breaking down the cleaned text into individual words, a process known as tokenization. Following this, a set of stopwords specific to the English language is created to filter out commonly occurring but less meaningful words. These stopwords are then removed from the tokenized text, streamlining the data for more focused analysis. Additionally, a lemmatizer is employed to transform words into their base or root forms, aiding in standardizing and simplifying the vocabulary. This sequence of operations collectively refines the text data, making it more suitable for subsequent tasks such as natural language processing or machine learning models.
### Word Embedding 
###  1-Word2Vec Embedding
#### After preparing the text data through tokenization, stopword removal, and lemmatization, a Word2Vec model is trained using the Continuous Bag of Words (CBOW) approach with the help of the Gensim library. This model learns distributed representations of words based on their context in sentences from the 'tokens' column of the DataFrame. Each word is embedded in a vector space of dimensionality 100, capturing semantic relationships between words. Subsequently, a function is defined to generate Word2Vec embeddings for each answer in the DataFrame by averaging the Word2Vec vectors of individual tokens present in the answer. This step creates dense numerical representations of text, enabling further analysis such as similarity calculations or feeding into machine learning models for tasks like text classification or clustering.
###  2-Bag of Words (BoW)
#### In this process, the tokenized text data is first converted back into string format by joining tokens with spaces. This step is necessary for the Bag of Words (BoW) model, which requires string input. A CountVectorizer from the sklearn library is then used to transform these strings into a BoW representation. The CountVectorizer generates a sparse matrix where each row corresponds to a text entry and each column represents a unique word from the entire corpus, with cell values indicating word frequencies. This sparse matrix is subsequently converted into a DataFrame for better readability and easier manipulation, where each column name represents a word from the vocabulary and each row corresponds to the BoW features of a specific text entry.
###  3-TF-IDF
#### In this process, the text data, which has been tokenized, cleaned, and reassembled into strings, is transformed using the TF-IDF (Term Frequency-Inverse Document Frequency) approach. A TfidfVectorizer from the sklearn library is employed to convert the processed text into a TF-IDF matrix. This matrix quantifies the importance of each word in a document relative to the entire corpus, balancing term frequency with inverse document frequency to mitigate the influence of common but less informative words. The resulting sparse matrix is then converted into a DataFrame for improved readability, where each column represents a unique word from the vocabulary and each row corresponds to the TF-IDF features of a specific text entry. 
### Model Training and Evaluation
###  1-SVR
<img width="608" alt="svm" src="https://github.com/WidadEs/Atelier3/assets/118807169/5e65783d-eeb3-47a9-8c23-5497e0436c65">

###  2-Naive Bayes
<img width="480" alt="nb" src="https://github.com/WidadEs/Atelier3/assets/118807169/70ff0d1c-ebfb-42cf-9b02-3744e9075c08">

###  3-Lineaer Regression
<img width="469" alt="lr" src="https://github.com/WidadEs/Atelier3/assets/118807169/47d08753-7ab2-486e-8fad-044c4c140c41">

### Decision Tree
<img width="467" alt="dt" src="https://github.com/WidadEs/Atelier3/assets/118807169/2e761ba6-627a-4369-82ac-e297075dd5ee">

### Comparison of model performance
<img width="616" alt="m" src="https://github.com/WidadEs/Atelier3/assets/118807169/0bacc2a2-28be-4f6e-ac35-7921fd9f5e35">

#### In conclusion, the low error values and effective relationship capture make Linear Regression the preferred choice for predictive modeling on this dataset. However, it's essential to consider other factors such as model assumptions, computational efficiency, and interpretability when selecting the final model.

## Part2 : Language Modeling / Classification
<img width="562" alt="cl" src="https://github.com/WidadEs/Atelier3/assets/118807169/196db195-6c12-470e-8aa6-c7afa470dbc4">

#### Logistic Regression and SVM demonstrate superior accuracy and classification metrics compared to Naive Bayes and AdaBoost. Despite this, AdaBoost and Naive Bayes offer valuable insights, especially in classifying majority classes. When selecting the optimal model for deployment, it is crucial to consider task-specific requirements and the balance between precision, recall, and accuracy.




