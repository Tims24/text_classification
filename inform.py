import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import nltk
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

submission = pd.DataFrame()

train = train.dropna()
test = test.dropna()

train['text'] = train['text'].astype('str')
test['text'] = test['text'].astype('str')

map_label = {'FAQ - тарифы и услуги': 0,
             'мобильная связь - тарифы': 1,
             'Мобильный интернет': 2,
             'FAQ - интернет': 3,
             'тарифы - подбор': 4,
             'Баланс': 5,
             'Мобильные услуги': 6,
             'Оплата': 7,
             'Личный кабинет': 8,
             'SIM-карта и номер': 9,
             'Роуминг': 10,
             'запрос обратной связи': 11,
             'Устройства': 12,
             'мобильная связь - зона обслуживания': 13}

map_label_back = {0: 'FAQ - тарифы и услуги',
                  1: 'мобильная связь - тарифы',
                  2: 'Мобильный интернет',
                  3: 'FAQ - интернет',
                  4: 'тарифы - подбор',
                  5: 'Баланс',
                  6: 'Мобильные услуги',
                  7: 'Оплата',
                  8: 'Личный кабинет',
                  9: 'SIM-карта и номер',
                  10: 'Роуминг',
                  11: 'запрос обратной связи',
                  12: 'Устройства',
                  13: 'мобильная связь - зона обслуживания'}

train['label'] = train['label'].map(map_label)


def Vectorize(vec, X_train, X_test):
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    print('Vectorization complete.\n')

    return X_train_vec, X_test_vec



param_rf = {'n_estimators': [1000]}

classifier = RandomForestClassifier()


def main():
    print('Data cleaning in progress...')

    train['text_clean'] = train['text'].str.lower()
    test['text_clean'] = test['text'].str.lower()

    train['text_clean'] = train['text_clean'].apply(nltk.word_tokenize)
    test['text_clean'] = test['text_clean'].apply(nltk.word_tokenize)
    print('Tokenization complete.')

    stop_words = set(nltk.corpus.stopwords.words("russian"))
    train['text_clean'] = train['text_clean'].apply(lambda x: [item for item in x if item not in stop_words])
    test['text_clean'] = test['text_clean'].apply(lambda x: [item for item in x if item not in stop_words])
    print('Stop words removed.')

    lem = nltk.stem.wordnet.WordNetLemmatizer()
    train['text_clean'] = train['text_clean'].apply(lambda x: [lem.lemmatize(item, pos='v') for item in x])
    test['text_clean'] = test['text_clean'].apply(lambda x: [lem.lemmatize(item, pos='v') for item in x])
    print('Lemmatization complete.\nData cleaning complete.\n')

    X_train, X_test, y_train, y_test = train_test_split(train['text_clean'], train['label'], test_size=0.2,
                                                        shuffle=True)
    X_train = X_train.apply(lambda x: ' '.join(x))
    X_test = X_test.apply(lambda x: ' '.join(x))
    test1 = test['text_clean'].apply(lambda x: ' '.join(x))
    X_train_vec, X_test_vec = Vectorize(TfidfVectorizer(max_features=8500), X_train, X_test)
    test_vec = TfidfVectorizer(max_features=8500).fit_transform(test1)

    print('Fitting in progress')
    grid_clf_rf = GridSearchCV(classifier, param_grid=param_rf, cv=5, n_jobs=4)
    grid_clf_rf.fit(X_train_vec, y_train)
    best_grid_clf_rf = grid_clf_rf.best_estimator_
    predict_grid = best_grid_clf_rf.predict(X_test_vec)
    print('Fitting finished')
    predict_clf_test = best_grid_clf_rf.predict(test_vec)
    submission['id'] = submission.index
    submission['label'] = predict_clf_test
    submission['label'] = submission['label'].map(map_label_back)
    submission.to_csv(r'submission.csv')


    print(confusion_matrix(y_test, predict_grid))
    print(classification_report(y_test, predict_grid))
    print(accuracy_score(y_test, predict_grid))

    with open('prediction.txt', 'w') as f:
        f.write('prediction\n')
        for i in predict_clf_test:
            f.write(str(i) + '\n')
    with open('best_grid_clf_rf.pickle', 'wb') as f:
        pickle.dump(best_grid_clf_rf, f)


if __name__ == '__main__':
    main()
