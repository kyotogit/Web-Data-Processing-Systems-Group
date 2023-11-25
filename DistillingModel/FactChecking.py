from transformers import pipeline
import requests
from bs4 import BeautifulSoup


# Parse the generated text and extract the triplets
def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'Subject': subject.strip(), 'Relation': relation.strip(), 'Object': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'Subject': subject.strip(), 'Relation': relation.strip(), 'Object': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'Subject': subject.strip(), 'Relation': relation.strip(), 'Object': object_.strip()})
    return triplets


def fetch_wikidata(params):
    url = 'https://www.wikidata.org/w/api.php'
    try:
        return requests.get(url, params=params)
    except:
        return 'There was an error'


# Fetch the Wikidata ID of the entity
def get_entity_id(entity):
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'search': entity,
        'language': 'en',
        'type': 'item',
    }

    data = fetch_wikidata(params)
    data = data.json()
    # print(json.dumps(data, indent=5, ensure_ascii=False))

    entity_ids = []
    for item in data['search']:
        entity_ids.append(item['id'])

    return entity_ids


# Perform a SPARQL query to get the relation between two entities
def sparql_query(subj_id, obj_id):
    url = 'https://query.wikidata.org/sparql'
    query = """
    SELECT ?relationLabel
    WHERE {{
      wd:{} ?relation wd:{}.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """
    query = query.format(subj_id, obj_id)

    r = requests.get(url, params={'format': 'json', 'query': query})
    data = r.json()
    return data


triplet_extractor = pipeline(
    'text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large'
)

# prompt = 'Is Managua the capital of Nicaragua?'
# prompt = "Paris is capital of Nicaragua"
# prompt = 'What is the capital of China?'
# prompt = "The capital of China is ...?"
# extracted_answer = 'Beijing'
# correctness = ''


def check_fact(prompt, extracted_answer, correctness):
    # We need to use the tokenizer manually since we need special tokens.
    extracted_text = triplet_extractor.tokenizer.batch_decode(
        [triplet_extractor(prompt, return_tensors=True, return_text=False)[0]["generated_token_ids"]]
    )

    extracted_triplets = extract_triplets(extracted_text[0])
    # print(extracted_triplets)

    if extracted_answer == 'yes' or extracted_answer == 'no':
        for triplet in extracted_triplets:
            # print(triplet['Subject'], triplet['Relation'], triplet['Object'])
            subject_ids = get_entity_id(triplet['Subject'])
            object_ids = get_entity_id(triplet['Object'])
            # print(subject_ids)
            # print(object_ids)
            relation = sparql_query(subject_ids[0], object_ids[0])
            # print(relation)
            if len(relation['results']['bindings']) == 0:
                if extracted_answer == 'yes':
                    correctness = 'incorrect'
                elif extracted_answer == 'no':
                    correctness = 'correct'
            else:
                relation_label = relation['results']['bindings'][0]['relationLabel']['value']
                # print(relation_label)

                # scrape the relationLabel page to get the relation name
                url = relation_label
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")
                span_element = soup.find("span", class_="wikibase-title-label")
                relation_name = span_element.text
                if relation_name in triplet['Relation'] and extracted_answer == 'yes':
                    correctness = 'correct'
                    break
                elif relation_name in triplet['Relation'] and extracted_answer == 'no':
                    correctness = 'incorrect'
                    break

        if correctness == '' and extracted_answer == 'yes':
            correctness = 'incorrect'
        elif correctness == '' and extracted_answer == 'no':
            correctness = 'correct'

        # check if the prompt contains negation
        negations = ['is not', 'are not', 'isn\'t', 'aren\'t', 'was not', 'were not', 'wasn\'t', 'weren\'t']
        for negation in negations:
            if negation in prompt:
                if correctness == 'correct':
                    correctness = 'incorrect'
                elif correctness == 'incorrect':
                    correctness = 'correct'
                break
    else:
        for triplet in extracted_triplets:
            # print(triplet['Subject'], triplet['Relation'], triplet['Object'])
            if len(triplet['Subject']) > 0:
                subject_ids = get_entity_id(triplet['Subject'])
                if len(subject_ids) == 0:
                    if len(triplet['Object']) > 0:
                        subject_ids = get_entity_id(triplet['Object'])
                        if len(subject_ids) == 0:
                            continue
                    elif len(triplet['Object']) == 0:
                        continue
            elif len(triplet['Subject']) == 0:
                if len(triplet['Object']) > 0:
                    subject_ids = get_entity_id(triplet['Object'])
                    if len(subject_ids) == 0:
                        continue
                elif len(triplet['Object']) == 0:
                    continue

            if len(extracted_answer) > 0:
                object_ids = get_entity_id(extracted_answer)
                if len(object_ids) == 0:
                    continue
            elif len(extracted_answer) == 0:
                correctness = 'Unable to make a judgment based on the extracted answer'
                break
            # print(subject_ids)
            # print(object_ids)
            relation = sparql_query(subject_ids[0], object_ids[0])
            # print(relation)
            if len(relation['results']['bindings']) == 0:
                correctness = 'incorrect'
            else:
                relation_label = relation['results']['bindings'][0]['relationLabel']['value']
                # print(relation_label)

                # scrape the relationLabel page to get the relation name
                url = relation_label
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")
                span_element = soup.find("span", class_="wikibase-title-label")
                relation_name = span_element.text
                if relation_name in triplet['Relation']:
                    correctness = 'correct'
                    break

    if correctness == '':
        correctness = 'incorrect'

    return correctness


# print(check_fact(prompt, extracted_answer, correctness))
