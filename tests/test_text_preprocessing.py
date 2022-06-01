import pytest
from src.inference_api.modules.bow_vectorizer import BowVectorizer
from src.inference_api.modules.text_preparer import TextPreparer

TEST_WORDS_TO_INDEX = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}


@pytest.mark.parametrize('input_str, result_vector',
                         [('hi how are you', [1, 1, 0, 1]),
                          ('hello world', [0, 0, 0, 0]),
                          ('you and me', [0, 1, 1, 0])])
def test_my_bag_of_words(input_str, result_vector):
    bow_vectorizer = BowVectorizer(TEST_WORDS_TO_INDEX, len(TEST_WORDS_TO_INDEX))
    transformed_data = bow_vectorizer.my_bag_of_words(input_str).tolist()

    assert transformed_data == result_vector

@pytest.mark.parametrize('origin_description, result_description', [
    ("SQL Server - any equivalent of Excel's CHOOSE function?", "sql server equivalent excels choose function"),
    ("How to free c++ memory vector<int> * arr?", "free c++ memory vectorint arr"),
    ("Hello this is not a question", "hello question")])
def test_text_prepare(origin_description, result_description):
    text_preparer = TextPreparer()
    prepared_data = text_preparer.text_prepare(origin_description)

    assert prepared_data == result_description
