import difflib
import os
from src.data_processing import load_data, normalize_khmer, khmer_word, flat_khmer_word
from src.predict import transliterate_variants
from config import BASE_DIR


all_pronunciations = load_data(
    os.path.join(BASE_DIR, "data", "raw", "eng_khm_data.txt")
)


def find_similar(input_str, pronunciations_data=all_pronunciations, max_results=5):
    """
    Find similar pronunciations for a given input string from a list of pronunciations.

    Args:
        input_str (str): The input string to find similar pronunciations for.
        pronunciations_data (list): A list of tuples where each tuple contains a pronunciation and its corresponding Khmer word.
                                    Defaults to `all_pronunciations`.
        max_results (int): The maximum number of unique Khmer words to return. Defaults to 5.

    Returns:
        list: A list of unique Khmer words that are similar to the input string, limited to `max_results` entries.

    Example:
        >>> all_pronunciations = [('som', 'សុំ'), ('som', 'សំ'), ('som', 'សុំ'), ('som', 'សំ'), ('som', 'សុំ')]
        >>> find_similar('som', all_pronunciations, max_results=2)
        ['សុំ', 'សំ']
    """
    input_str = input_str.lower()
    # Get top 10 matches (to account for duplicates)
    matches = difflib.get_close_matches(
        input_str, [p[0] for p in pronunciations_data], n=10, cutoff=0.5
    )
    # Deduplicate Khmer words while preserving order
    seen = set()
    results = []
    for pron in matches:
        for p, khmer in pronunciations_data:
            if p == pron and khmer not in seen:
                seen.add(khmer)
                results.append(khmer)
                if len(results) == max_results:
                    return results
                break  # Move to next match
    return results


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def prepare_tfidf(pronunciations_data=all_pronunciations):
    """
    Prepare the TF-IDF vectorizer and matrix for the given pronunciations data.

    Args:
        pronunciations_data (list): A list of tuples where each tuple contains a pronunciation and its corresponding Khmer word.
                                    Defaults to `all_pronunciations`.

    Returns:
        tuple: A tuple containing the TF-IDF vectorizer, the TF-IDF matrix, and the list of Khmer words.

    Example:
        >>> pronunciations_data = [('som', 'សុំ'), ('som', 'សំ'), ('som', 'សុំ')]
        >>> vectorizer, tfidf_matrix, khmer_words = prepare_tfidf(pronunciations_data)
        >>> type(vectorizer)
        <class 'sklearn.feature_extraction.text.TfidfVectorizer'>
        >>> tfidf_matrix.shape
        (3, 6)
        >>> khmer_words
        ['សុំ', 'សំ', 'សុំ']
    """
    pronunciations = [p[0] for p in pronunciations_data]
    khmer_words = [p[1] for p in pronunciations_data]

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    tfidf_matrix = vectorizer.fit_transform(pronunciations)

    return vectorizer, tfidf_matrix, khmer_words


_vectorizer, _tfidf_matrix, _khmer_words = prepare_tfidf()


def find_similar_tfidf(
    input_str,
    vectorizer=_vectorizer,
    tfidf_matrix=_tfidf_matrix,
    khmer_words=_khmer_words,
    max_results=5,
):
    """
    Find similar Khmer words based on TF-IDF cosine similarity.
    This function takes an input string, transforms it using the provided TF-IDF vectorizer,
    and computes the cosine similarity between the input vector and a precomputed TF-IDF matrix.
    It returns a list of the most similar Khmer words, deduplicated and limited to a specified number of results.

    Args:
        input_str (str): The input string to find similar words for.
        vectorizer (sklearn.feature_extraction.text.TfidfVectorizer): The TF-IDF vectorizer used to transform the input string.
        tfidf_matrix (scipy.sparse.csr.csr_matrix): The precomputed TF-IDF matrix of the corpus.
        khmer_words (list): A list of Khmer words corresponding to the rows of the TF-IDF matrix.
        max_results (int, optional): The maximum number of similar words to return. Defaults to 5.

    Returns:
        list: A list of the most similar Khmer words, limited to `max_results` entries.
    """
    input_vec = vectorizer.transform([input_str.lower()])
    similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()
    # Get top indices
    top_indices = similarities.argsort()[::-1][:10]  # Top 10 for dedup
    seen = set()
    results = []
    for idx in top_indices:
        khmer = khmer_words[idx]
        if khmer not in seen:
            seen.add(khmer)
            results.append(khmer)
            if len(results) == max_results:
                break
    return results


correct_khmer_normalized = flat_khmer_word(
    os.path.join(BASE_DIR, "data", "raw", "khmer_words.txt")
)


from collections import defaultdict
from rapidfuzz import fuzz, process


def predict_last_result(text, num_results=3):
    """
    Predicts the most likely transliterations of the given text.

    This function generates multiple transliteration variants of the input text
    and uses fuzzy matching to score and rank these variants against a list of
    correct Khmer normalized words. The top `num_results` transliterations with
    the highest scores are returned.

    Args:
        text (str): The input text to be transliterated.
        num_results (int, optional): The number of top results to return. Defaults to 3.

    Returns:
        list: A list of the top `num_results` transliterations with the highest scores.

    Example:
        >>> predict_last_result("example text", num_results=3)
        ['transliteration1', 'transliteration2', 'transliteration3']
    """
    # Step 1: get output from model and create suggestions word
    seq2seq_outputs = transliterate_variants(text)
    all_suggestions = []
    for generated_word in seq2seq_outputs:
        # Use RapidFuzz to find matches with scores
        matches = process.extract(
            generated_word,
            correct_khmer_normalized,
            scorer=fuzz.ratio,  # Levenshtein ratio
            limit=3,  # Top 5 per seq2seq output
        )
        all_suggestions.extend(matches)  # [(word, score, index), ...]

    # Step 2: Deduplicate and keep highest score per word
    word_scores = defaultdict(int)
    for word, score, _ in all_suggestions:
        if score > word_scores[word]:
            word_scores[word] = score

    # Step 3: Sort by score (descending) and pick top 5
    sorted_words = sorted(word_scores.items(), key=lambda x: -x[1])
    final_results = [word for word, score in sorted_words[:num_results]]

    return final_results
