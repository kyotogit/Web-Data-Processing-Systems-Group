from llama_cpp import Llama
import spacy
# packages used for TASK 1
import wikipedia
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# packages used for TASK 2
from transformers import AutoTokenizer, DistilBertForQuestionAnswering
import torch
import re
# packages used for TASK 3
import FactChecking

# load the large language model file and define n_ctx manually to permit larger contexts
llm = Llama(model_path="/Users/erynnbai/Downloads/llama-2-13b-chat.Q4_K_M.gguf", n_ctx=256)

# create a text prompt
prompt = input("Type your question (for instance: \"The capital of Italy is ...?\") and type ENTER to finish:\n")
print("Computing the answer (can take some time)...")

# generate a response, set max_tokens to 0 to remove the response size limit
output = llm(prompt, max_tokens=0)
response = output["choices"][0]["text"]


# TASK 0: preprocessing
nlp = spacy.load("en_core_web_sm")
prompt_doc = nlp(prompt.strip())
response_doc = nlp(response.strip())

prompt_mentions = [str(mention) for mention in prompt_doc.ents]
response_mentions = [str(mention) for mention in response_doc.ents]

entities = {}
for item in prompt_mentions:
    if item not in entities.keys():
        entities[item] = ''
for item in response_mentions:
    if item not in entities.keys():
        entities[item] = ''
print(entities)


# TASK 1: entity linking
# step 1: generate candidate entities
# NB: Don't forget to add NIL as a special candidate entity to address Unlinkable Mention Prediction
# step 2: use vector space model(VSM) to do candidate entity ranking
def get_candidate(mention, max_num):
    candidate_list = wikipedia.search(mention, results=max_num)
    return candidate_list


def vector_space_model(mention, candidate):
    # create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    # fit and transform the documents
    tfidf_matrix = vectorizer.fit_transform([mention, candidate])
    # convert the matrix to an array for easier indexing
    tfidf_matrix_array = tfidf_matrix.toarray()
    # calculate cosine similarity between the two documents
    similarity = cosine_similarity(tfidf_matrix_array)[0, 1]
    return similarity


for mention in entities.keys():
    candidates = get_candidate(mention, 9)
    # add NIL as a special candidate entity
    candidates.insert(0, "NIL")
    print(candidates)

    similarity_dict = {}
    for candidate in candidates:
        similarity_dict[candidate] = vector_space_model(mention, candidate)
        similarity_dict_sorted = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)

    top_candidate = similarity_dict_sorted[0][0]
    if top_candidate == "NIL":
        entities[mention] = "\"Unlinkable\""
    else:
        try:
            top_candidate_page = wikipedia.page(top_candidate, auto_suggest=False)
            entities[mention] = top_candidate_page.url
        except:
            try:
                url = "https://en.wikipedia.org/wiki/" + top_candidate
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")

                target_div_class = "mw-content-ltr mw-parser-output"
                target_div = soup.find("div", class_=target_div_class)
                ul_tag = target_div.find("ul")
                first_li_tag = ul_tag.find("li")
                a_tag = first_li_tag.find("a")
                title = a_tag.get("title")
                top_candidate_page = wikipedia.page(title, auto_suggest=False)
                entities[mention] = top_candidate_page.url
            except:
                entities[mention] = "\"Unlinkable\""


# TASK 2: extract answer from llama2
# fine-tuning DistilBERT to implement the downstream task of extracting the answer from llama2's response
# the relevant code for fine-tuning DistilBERT is located in AnswerExtraction.py
save_dir = '/Users/erynnbai/PycharmProjects/DistillingModel/distilbert-base-uncased-answer-extraction'
tokenizer = AutoTokenizer.from_pretrained(save_dir, local_files_only=True)
model = DistilBertForQuestionAnswering.from_pretrained(save_dir, local_files_only=True)


def extracting_answer(prompt, response):
    inputs = tokenizer(prompt, response, return_tensors="pt")
    input_ids = inputs.input_ids.tolist()[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    with torch.no_grad():
        outputs = model(**inputs)

    # reconstructing the extracted_answer
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()
    if answer_end_index >= answer_start_index:
        answer = tokens[answer_start_index]
        for i in range(answer_start_index + 1, answer_end_index + 1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]

    if answer.startswith("[CLS]"):
        answer = "Failed to extract the answer"

    return answer


extracted_answer = extracting_answer(prompt, response)

pattern_yes = re.compile(r'\byes\s*(?:[.,;?!])', re.IGNORECASE)
pattern_no = re.compile(r'\bno\s*(?:[.,;?!])', re.IGNORECASE)

condition = 0
if pattern_yes.search(response):
    extracted_answer = "yes"
    condition = 1
elif pattern_no.search(response):
    extracted_answer = "no"
    condition = 1
if condition == 0:
    for mention in entities.keys():
        if mention.lower() in extracted_answer:
            extracted_answer = mention
            extracted_answer_linking = entities[mention]
if extracted_answer == "Failed to extract the answer":
    condition = 1


# TASK 3: judge whether the answer given by llama2 is correct
correctness = ''  # correct or incorrect
correctness = FactChecking.check_fact(prompt, extracted_answer, correctness)


# display the output
print("COMPLETION:")
print(f'Text returned by the language model: "{response.strip()}"')
if condition == 1:
    print(f'Extracted answer: "{extracted_answer}"')
else:
    print(f'Extracted answer: {extracted_answer_linking}')
print(f'Correctness of the answer: "{correctness}"')
print("Entities extracted:")
for mention, entity_linking in entities.items():
    print(f'{mention}\t{entity_linking}')
