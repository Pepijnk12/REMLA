"""
Load preprocessed data and generate model
"""
from joblib import load, dump
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def print_words_for_tag(classifier, tag, tags_classes, index_to_words, all_words):
    """
        classifier: trained classifier
        tag: particular tag
        tags_classes: a list of classes names from MultiLabelBinarizer
        index_to_words: index_to_words transformation
        all_words: all words in the dictionary

        return nothing, just print top 5 positive and top 5 negative words for current tag
    """
    print(f'Tag:\t{tag}')

    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator.

    model = classifier.estimators_[tags_classes.index(tag)]
    top_positive_words = [index_to_words[x] for x in model.coef_.argsort().tolist()[0][-5:]]
    top_negative_words = [index_to_words[x] for x in model.coef_.argsort().tolist()[0][:5]]

    print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
    print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))


def train_classifier(X_train, y_train, penalty='l1', C=1):
    """
      X_train, y_train — training data

      return: trained classifier
    """

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver='liblinear')
    clf = OneVsRestClassifier(clf)
    clf.fit(X_train, y_train)

    return clf


def print_evaluation_scores(y_val, predicted):
    print('Accuracy score: ', accuracy_score(y_val, predicted))
    print('F1 score: ', f1_score(y_val, predicted, average='weighted'))
    print('Average precision score: ', average_precision_score(y_val, predicted, average='macro'))


def main():
    X_train_mybag = load('output/preprocessed_x_train_mybag.joblib')
    X_val_mybag = load('output/preprocessed_x_val_mybag.joblib')
    X_test_mybag = load('output/preprocessed_x_test_mybag.joblib')

    X_train_tfidf = load('output/preprocessed_x_train_tfidf.joblib')
    X_val_tfidf = load('output/preprocessed_x_val_tfidf.joblib')
    X_test_tfidf = load('output/preprocessed_x_test_tfidf.joblib')
    tfidf_vocab = load('output/tfidf_vocab.joblib')

    y_train = load('output/y_train.joblib')
    y_val = load('output/y_val.joblib')
    tags_counts = load('output/tags_counts.joblib')
    WORDS_TO_INDEX = load('output/WORDS_TO_INDEX.joblib')

    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    classifier_mybag = train_classifier(X_train_mybag, y_train)
    classifier_tfidf = train_classifier(X_train_tfidf, y_train)

    y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
    y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

    y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
    y_val_inversed = mlb.inverse_transform(y_val)

    print('======= Bag-of-words ========')
    print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
    roc_auc_score_bow = roc_auc_score(y_val, y_val_predicted_scores_mybag, multi_class='ovo')
    print(f'ROC AOC score: {roc_auc_score_bow}')
    print('======= Tfidf ========')
    print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)
    roc_auc_score_tfidf = roc_auc_score(y_val, y_val_predicted_scores_mybag, multi_class='ovo')
    print(f'ROC AOC score: {roc_auc_score_tfidf}')

    # L2-regularization and coefficient 10 make best performance
    classifier_tfidf = train_classifier(X_train_tfidf, y_train, penalty='l2', C=10)
    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

    test_predictions = classifier_tfidf.predict(X_test_tfidf)
    test_pred_inversed = mlb.inverse_transform(test_predictions)

    test_predictions_for_submission = '\n'.join(
        '%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))
    with open('output/test_predict_result.tsv', 'w') as test_pred_result_file:
        test_pred_result_file.write(test_predictions_for_submission)

    tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}
    ALL_WORDS = WORDS_TO_INDEX.keys()
    print_words_for_tag(classifier_tfidf, 'c', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
    print_words_for_tag(classifier_tfidf, 'c++', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
    print_words_for_tag(classifier_tfidf, 'linux', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)

    dump(classifier_tfidf, 'output/model_tfidf.joblib')
    dump(classifier_mybag, 'output/model_mybag.joblib')


if __name__ == "__main__":
    main()
