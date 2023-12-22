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
import string
# packages used for TASK 3
import FactChecking
import EntityLinking

# load the large language model file and define n_ctx manually to permit larger contexts
llm = Llama(model_path="D:/chromefiles/llama-2-13b-chat.Q4_K_M.gguf", n_ctx=256)

# create a text prompt
# prompt = input("Type your question (for instance: \"The capital of Italy is ...\") and type ENTER to finish:\n")
# print("Computing the answer (can take some time)...")
# with open('test_filtered_processed.txt', 'r', encoding='utf-8') as txt_file:
with open('./data/example_input1.txt', 'r', encoding='utf-8') as txt_file:
    lines = txt_file.readlines()

# 预处理并存储到序列中
processed_questions = []
cut_pard_list = []
for line in lines:
    # 切割每行的字符串，保留 "Question: " 后面的部分
    question_start = 13
    question = line[question_start:].strip()
    cut_part = line[:question_start].strip()
    # question = line[13:].strip()
    processed_questions.append(question)
    cut_pard_list.append(cut_part)
# print(processed_questions)


# TASK 1: entity linking
# step 1: generate candidate entities
# NB: Don't forget to add NIL as a special candidate entity to address Unlinkable Mention Prediction
# step 2: use vector space model(VSM) to do candidate entity ranking
def remove_punctuation(text):
    # 创建一个翻译表，将标点符号映射为 None
    translator = str.maketrans("", "", string.punctuation)
    # 应用翻译表
    text_without_punctuation = text.translate(translator)
    return text_without_punctuation

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

# Task 2: function definition
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
    # print("answer: " + answer)
    if answer.startswith("[CLS]"):
        answer = "Failed to extract the answer"

    return answer

##############START
# generate a response, set max_tokens to 0 to remove the response size limit
# output
output_file_name = 'file_output.txt'
open(output_file_name, 'a', encoding='utf-8')
num = 0
for prompt in processed_questions:
    # print("prompt: ")
    # print(prompt)
    output = llm(prompt, max_tokens=0)
    response = output["choices"][0]["text"]
    # delete quotations
    response = remove_punctuation(response)


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
    # print(entities)

    for mention in entities.keys():
        candidates = get_candidate(mention, 9)
        # add NIL as a special candidate entity
        candidates.insert(0, "NIL")
        # print(candidates)

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
    save_dir = '.\distilbert-base-uncased-answer-extraction'
    tokenizer = AutoTokenizer.from_pretrained(save_dir, local_files_only=True)
    model = DistilBertForQuestionAnswering.from_pretrained(save_dir, local_files_only=True)

    extracted_answer = extracting_answer(prompt, response)

    pattern_yes = re.compile(r'\byes\s*(?:[.,;?!])', re.IGNORECASE)
    pattern_no = re.compile(r'\bno\s*(?:[.,;?!])', re.IGNORECASE)

    condition = 0
    # if pattern_yes.search(response):
    if extracted_answer == "yes":
        condition = 1
    # elif pattern_no.search(response):
    if extracted_answer == "no":
        # extracted_answer = "no"
        condition = 1
    if condition == 0:
        for mention in entities.keys():
            if mention.lower() in extracted_answer:
                extracted_answer_ = entities[mention].replace("https://en.wikipedia.org/wiki/", "")
                # print(extracted_answer_)
                extracted_answer = extracted_answer_.replace("_", " ")
                # print(extracted_answer)
                extracted_answer_linking = entities[mention]
        # to be modified
        # condition = 1
    if extracted_answer == "Failed to extract the answer":
        condition = 1


    # TASK 3: judge whether the answer given by llama2 is correct
    correctness = ''  # correct or incorrect
    correctness = FactChecking.check_fact(prompt, extracted_answer, correctness)

    output_list = []
    # display the output
    # print("COMPLETION:")
    print(f'R"{response.strip()}"')
    output_list.append(cut_pard_list[num]+"\t"+f'R"{response.strip()}"')
    if condition == 1:
        print(f'A"{extracted_answer}"')
        output_list.append(cut_pard_list[num]+"\t"+f'A"{extracted_answer}"')
    else:
        print(f'A"{extracted_answer_linking}"')
        output_list.append(cut_pard_list[num]+"\t"+f'A"{extracted_answer_linking}"')
    print(f'C"{correctness}"')
    output_list.append(cut_pard_list[num] +"\t"+ f'C"{correctness}"')
    # print("Entities extracted:")
    for mention, entity_linking in entities.items():
        # print(f'E"{mention}\t{entity_linking}"')
        print(f'E"{entity_linking}"')
        output_list.append(cut_pard_list[num] +"\t"+ f'E"{entity_linking}"')

    # write into txt
    with open(output_file_name, 'a', encoding='utf-8') as output_file:
        for line in output_list:
            output_file.write(line + '\n')

    num = num + 1
