import nltk
import re
import requests
import os
from bs4 import BeautifulSoup
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set environment for Stanford NER
java_path = "D:/develop/jdk8/bin/java.exe"
os.environ['JAVAHOME'] = java_path

# Set paths to NER JAR and model 
ner_path = "./model/stanford-ner.jar"
model_path = "./model/english.muc.7class.distsim.crf.ser.gz"
ner_tagger = StanfordNERTagger(model_path, ner_path, encoding='utf-8')

def create_wikipedia_links(classified_text):
    links = []
    current_entity = []
    for word, tag in classified_text:
        if tag != 'O':
            current_entity.append(word)
        elif current_entity:
            # Join the words in the entity and clean it for URL
            entity_name = "_".join(current_entity)
            entity_name = re.sub(r'\W+', '_', entity_name)  # Replace non-alphanumeric characters with underscore
            wiki_url = f"https://en.wikipedia.org/wiki/{entity_name}"
            links.append((entity_name.replace('_', ' '), wiki_url))
            current_entity = []
    return links

def get_wikipedia_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    if 'disambiguation' in soup.find('body')['class']:
        return [li.get_text() for li in soup.find_all('li')]
    else:
        return [soup.find('p').get_text()]

def calculate_similarity(original_text, texts):
    documents = [original_text] + texts
    vectorizer = TfidfVectorizer().fit(documents)
    vectors = vectorizer.transform(documents)
    similarity_scores = cosine_similarity(vectors[0:1], vectors[1:])
    return similarity_scores.flatten()

def main():
    # Prompt and text
    Q = "what's the capital city of Netherlands?"
    text = "The capital city of the Netherlands is Amsterdam."

    # NER for text
    tokenized_text = word_tokenize(text)
    classified_text = ner_tagger.tag(tokenized_text)

    # Create Wikipedia links
    wikipedia_links = create_wikipedia_links(classified_text)

    for entity, link in wikipedia_links:
        print(f"Entity: {entity}, Link: {link}")

        # Get content from Wikipedia
        content_texts = get_wikipedia_content(link)

        # Calculate similarity
        similarity_scores = calculate_similarity(text, content_texts)

        # Select best match based on similarity
        best_match_index = similarity_scores.argmax()
        best_match_content = content_texts[best_match_index]

        print(f"Best Match for '{entity}': {best_match_content}")

if __name__ == "__main__":
    main()
