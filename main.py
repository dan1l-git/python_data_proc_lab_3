import re
import os
import ssl
import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
from uk_stemmer import UkStemmer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')


def task_1():
    print("\n--- Task 1: Regular Expressions (Basic Text Analysis) ---")
    text = (
        "This is the first sentence. Apples are very tasty and healthy. "
        "Python is a great programming language. "
        "Elephants are the largest land animals. "
        "I need to replace one word here."
    )

    # a. Number of words in the text
    words = re.findall(r'\b\w+\b', text)
    print(f"a. Total words: {len(words)}")

    # b. Words starting with a vowel and their count
    vowel_words = re.findall(r'\b[aeiouAEIOU]\w*', text)
    print(f"b. Words starting with a vowel ({len(vowel_words)}): {vowel_words}")

    # c. Words starting with a consonant
    consonant_words = re.findall(r'\b[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z]\w*', text)
    print(f"c. Words starting with a consonant: {consonant_words}")

    # d. Select 3 words and find their positions
    target_words = ['Python', 'healthy', 'animals']
    positions = {word: [m.start() for m in re.finditer(rf'\b{word}\b', text)] for word in target_words}
    print(f"d. Word positions: {positions}")

    # e. Replace a word with the surname
    modified_text = re.sub(r'\bword\b', 'Havryliuk', text)
    print(f"e. Modified text: {modified_text}")


def task_2():
    print("\n--- Task 2: String and List Manipulations ---")
    text = "apple banana cherry orange grapes melon"
    words = text.split()

    # List of characters without spaces
    char_list = list(text.replace(" ", ""))
    print(f"Character list: {char_list}")

    # List of the first two letters of each word
    first_two_letters = [word[:2] for word in words]
    print(f"First two letters: {first_two_letters}")

    # New list excluding 'a' and 'b' (Cyrillic 'а', 'б' or Latin 'a', 'b')
    filtered_chars = [char for char in char_list if char not in ['a', 'b', 'а', 'б']]
    print(f"Filtered characters (no 'a', 'b'): {filtered_chars}")


def task_3():
    print("\n--- Task 3: Extracting Programming Languages ---")
    text = "I love programming in Python and C++. Some people prefer Java or JavaScript, but Ruby is also nice."

    # Define a pattern for known programming languages
    languages_pattern = r'\b(Python|C\+\+|Java|JavaScript|Ruby|C#|PHP|Swift)\b'
    found_languages = re.findall(languages_pattern, text)
    print(f"Programming languages found: {found_languages}")


def task_4():
    print("\n--- Task 4: Extracting Email Domains ---")
    text = "John: john.doe@gmail.com, Alice: alice123@yahoo.com, Bob: bob.smith@kpi.ua"

    # Extract domains using regex
    domains = re.findall(r'@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text)
    print(f"Extracted email domains: {domains}")


def task_5():
    print("\n--- Task 5: Finding Numbers with Specific Digit Sequences ---")
    text = "Some numbers: 333334 333 123 2334 12558 5554 55555 90551 55"

    # Find numbers containing '5' exactly 2 to 3 times in a row
    # (?<!5) ensures no '5' before, (?!5) ensures no '5' after the sequence
    pattern = r'\b\d*(?<!5)5{2,3}(?!5)\d*\b'
    matched_numbers = re.findall(pattern, text)
    print(f"Numbers containing '55' or '555': {matched_numbers}")


def task_6():
    print("\n--- Task 6: Extracting Dates and Times ---")
    text = "Event started on 22.09.2026 at 12:27:35. Backup created on 01-10-2027 at 09:15:00."

    time_pattern = r'\b((\d{2}):\d{2}:\d{2})\b'
    date_pattern = r'\b(\d{2}[./-]\d{2}[./-](\d{4}))\b'

    time_matches = re.findall(time_pattern, text)
    date_matches = re.findall(date_pattern, text)

    time_list = [m[0] for m in time_matches]
    hour_list = [m[1] for m in time_matches]
    date_list = [m[0] for m in date_matches]
    year_list = [m[1] for m in date_matches]

    print("Times:", time_list)
    print("Dates:", date_list)
    print("Hours:", hour_list)
    print("Years:", year_list)


def task_7():
    print("\n--- Task 7: Ukrainian Postal Code Validation ---")
    test_codes = ["01001", "1234", "65000", "AB123", "999999"]

    # Ukrainian postal code is exactly 5 digits
    pattern = r'^\d{5}$'
    for code in test_codes:
        if re.fullmatch(pattern, code):
            print(f"'{code}' is a VALID postal code.")
        else:
            print(f"'{code}' is INVALID.")


def task_8():
    print("\n--- Task 8: NLTK Tokenization, Stop Words, and POS Tagging ---")
    text = (
        "Natural language processing is fascinating. Python is great for this task! "
        "We are learning how to tokenize sentences. "
        "Stop words should be removed carefully. "
        "This is the final sentence of this test."
    )

    # Tokenize by sentences
    sentences = sent_tokenize(text)
    print(f"Sentences: {sentences}")

    # Tokenize by words
    words = word_tokenize(text)

    # Remove punctuation (keep only alphanumeric words)
    words_no_punct = [w for w in words if w.isalnum()]

    # Remove stop words
    stop_words_eng = set(stopwords.words('english'))
    filtered_words = [w for w in words_no_punct if w.lower() not in stop_words_eng]
    print(f"Filtered words (no punct, no stop words): {filtered_words}")

    # POS tagging
    pos_tags = pos_tag(filtered_words)
    print(f"POS Tags: {pos_tags}")

    return sentences, filtered_words


def task_9(sentences, filtered_words):
    print("\n--- Task 9: Text Analytics ---")

    # Word count in each sentence
    for i, sentence in enumerate(sentences):
        word_count = len(re.findall(r'\b\w+\b', sentence))
        print(f"Sentence {i + 1} word count: {word_count}")

    # Determine the number of stop words in the text
    all_words = word_tokenize(" ".join(sentences))
    stop_words_eng = set(stopwords.words('english'))
    stop_words_count = sum(1 for w in all_words if w.lower() in stop_words_eng)
    print(f"Total stop words in text: {stop_words_count}")

    # Find the longest word
    longest_word = max(filtered_words, key=len)
    print(f"Longest word: {longest_word} ({len(longest_word)} chars)")

    # Count words with exactly 4 characters
    four_char_words = [w for w in all_words if len(w) == 4 and w.isalnum()]
    print(f"Words with 4 characters ({len(four_char_words)}): {four_char_words}")


def task_10(filtered_words):
    print("\n--- Task 10: Stemming vs Lemmatization (English) ---")

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    print(f"{'Original':<15} | {'Stemming (Porter)':<18} | {'Lemmatization (WordNet)':<20}")
    print("-" * 55)

    for word in filtered_words:
        stemmed = stemmer.stem(word)
        # using 'v' for verb to show better lemmatization results for test
        lemmatized = lemmatizer.lemmatize(word.lower(), pos='v')
        print(f"{word:<15} | {stemmed:<18} | {lemmatized:<20}")


def task_11():
    print("\n--- Task 11: Ukrainian NLP (Spacy & UkStemmer) ---")

    ukr_text = (
        "Штучний інтелект стрімко розвивається кожного дня. "
        "Програмісти використовують сучасні бібліотеки для обробки природної мови. "
        "Цей текст створено спеціально для перевірки роботи стемінгу та лематизації. "
        "Студенти успішно виконують лабораторні роботи в університеті."
    )

    # Create a custom stop words file if it doesn't exist
    stop_words_file = 'stop_words_ukr.txt'
    if not os.path.exists(stop_words_file):
        with open(stop_words_file, 'w', encoding='utf-8') as f:
            f.write("для\nта\nв\nна\nз\nдо\nце\nщо\nяк\nа\nі\nй")

    # Read custom stop words
    with open(stop_words_file, 'r', encoding='utf-8') as f:
        stop_words_ukr = set(f.read().splitlines())

    # Tokenization using Spacy
    try:
        nlp = spacy.load("uk_core_news_sm")
    except OSError:
        print("Model 'uk_core_news_sm' not found. Please run: python -m spacy download uk_core_news_sm")
        return

    doc = nlp(ukr_text)

    # Tokenize and remove punctuation
    words = [token.text for token in doc if not token.is_punct and not token.is_space]

    # Remove stop words
    filtered_words = [w for w in words if w.lower() not in stop_words_ukr]

    # Statistics
    print(f"Volume of original text (chars): {len(ukr_text)}")
    print(f"Number of punctuation marks: {sum(1 for token in doc if token.is_punct)}")
    print(f"Number of stop words found: {len(words) - len(filtered_words)}")
    print(f"Number of words in output text: {len(filtered_words)}")

    # Stemming (UkStemmer) and Lemmatization (Spacy)
    stemmer = UkStemmer()

    print("\nComparison for Ukrainian:")
    print(f"{'Original':<20} | {'Stemming (UkStemmer)':<20} | {'Lemmatization (Spacy)':<20}")
    print("-" * 65)

    # Retrieve lemmatized tokens directly from Spacy doc
    lemmas = [token.lemma_ for token in doc if
              not token.is_punct and not token.is_space and token.text.lower() not in stop_words_ukr]

    for i in range(len(filtered_words)):
        word = filtered_words[i]
        stemmed = stemmer.stem_word(word)
        lemmatized = lemmas[i]
        print(f"{word:<20} | {stemmed:<20} | {lemmatized:<20}")


if __name__ == "__main__":
    task_1()
    task_2()
    task_3()
    task_4()
    task_5()
    task_6()
    task_7()

    # Task 8 returns data needed for Task 9 and Task 10
    sentences_list, filtered_words_list = task_8()

    task_9(sentences_list, filtered_words_list)
    task_10(filtered_words_list)
    task_11()