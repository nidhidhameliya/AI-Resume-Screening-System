from src.preprocess import preprocess_text


def test_preprocess_removes_basic_stopwords_and_normalizes() -> None:
    text = "Experienced Python developer with Machine Learning skills."
    processed = preprocess_text(text)
    assert processed == "experienced python developer machine learning skills"
