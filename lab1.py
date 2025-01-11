import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

def download_nltk_resources():
    # Download necessary NLTK data files
    nltk.download('punkt')
    nltk.download('stopwords')

def main():
    # Download required NLTK resources
    download_nltk_resources()

    # Sample text
    text = "Natural Language Processing (NLP) is a fascinating field of artificial intelligence. NLP enables machines to understand, interpret, and generate human language."

    # Tokenize the text into words
    words = word_tokenize(text)
    print("Tokenized Words:", words)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    print("Filtered Words (without stopwords):", filtered_words)

    # Compute word frequency distribution
    freq_dist = FreqDist(filtered_words)
    print("Word Frequency Distribution:")
    for word, count in freq_dist.items():
        print(f"{word}: {count}")

if __name__ == "__main__":
    main()