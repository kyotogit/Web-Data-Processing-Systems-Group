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
    entity_id = data['search'][0]['id']

    return entity_id


# Perform a SPARQL query to get the relation between two entities
def sparql_query_wikidata(subj, obj):
    if len(subj) == 0 or len(obj) == 0:
        relation = 'No relation found'
    else:
        subj_id = get_entity_id(subj)
        obj_id = get_entity_id(obj)

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
        if len(data['results']['bindings']) == 0:
            relation = 'No relation found'
        else:
            relation_label = data['results']['bindings'][0]['relationLabel']['value']
            # scrape the relationLabel page to get the relation name
            url = relation_label
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            span_element = soup.find("span", class_="wikibase-title-label")
            relation = span_element.text

    return relation


def sparql_query_dbpedia(subj, obj):
    subj = subj.replace(' ', '_')
    obj = obj.replace(' ', '_')
    url = 'https://dbpedia.org/sparql/'
    query = """
    SELECT ?p
    WHERE {{
      <http://dbpedia.org/resource/{}> ?p <http://dbpedia.org/resource/{}>.
    }}
    """
    query = query.format(subj, obj)

    r = requests.get(url, params={'format': 'json', 'query': query})
    data = r.json()
    relation = ''
    if len(data['results']['bindings']) == 0:
        relation = 'No relation found'
    else:
        for i in range(len(data['results']['bindings'])):
            rlink = data['results']['bindings'][i]['p']['value']
            if 'wikiPageWikiLink' in rlink:
                continue
            elif 'property' in rlink:
                relation = rlink.replace('http://dbpedia.org/property/', '')
                break
            elif 'ontology' in rlink:
                relation = rlink.replace('http://dbpedia.org/ontology/', '')
                break

    return relation


# print(sparql_query_wikidata("Nicaragua", "Managua"))
# print(sparql_query_wikidata('Barack Obama', ''))
# print(sparql_query_dbpedia('Nicaragua', 'Managua'))

triplet_extractor = pipeline(
    'text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large'
)

prompt = 'Is Beijing the capital of China?'
# prompt = "Paris is capital of Nicaragua"
# prompt = 'What is the capital of China?'
# prompt = "The capital of China is ...?"
# prompt = "Obama was born in Hawaii."
extracted_answer = 'no'
correctness = ''


def check_fact(prompt, extracted_answer, correctness):
    # We need to use the tokenizer manually since we need special tokens.
    extracted_text = triplet_extractor.tokenizer.batch_decode(
        [triplet_extractor(prompt, return_tensors=True, return_text=False)[0]["generated_token_ids"]]
    )

    extracted_triplets = extract_triplets(extracted_text[0])
    print(extracted_triplets)

    if extracted_answer in ['yes', 'no']:
        for triplet in extracted_triplets:
            relations = []
            relations.append(sparql_query_wikidata(triplet['Subject'], triplet['Object']))
            relations.append(sparql_query_dbpedia(triplet['Subject'], triplet['Object']))
            print(relations)

            if triplet['Relation'] in relations and extracted_answer == 'yes':
                correctness = 'correct'
                break
            elif triplet['Relation'] in relations and extracted_answer == 'no':
                correctness = 'incorrect'
                break

        if correctness == '' and extracted_answer == 'yes':
            correctness = 'incorrect'
        elif correctness == '' and extracted_answer == 'no':
            correctness = 'correct'

        # check if the prompt contains negation
        negations = [
            'is not', 'are not', 'isn\'t', 'aren\'t',
            'was not', 'were not', 'wasn\'t', 'weren\'t',
            'does not', 'do not', 'doesn\'t', 'don\'t',
        ]
        for negation in negations:
            if negation in prompt:
                if correctness == 'correct':
                    correctness = 'incorrect'
                elif correctness == 'incorrect':
                    correctness = 'correct'
                break

    else:
        for triplet in extracted_triplets:
            relations = []
            relations.append(sparql_query_wikidata(triplet['Subject'], extracted_answer))
            relations.append(sparql_query_wikidata(extracted_answer, triplet['Object']))
            relations.append(sparql_query_dbpedia(triplet['Subject'], extracted_answer))
            relations.append(sparql_query_dbpedia(extracted_answer, triplet['Object']))
            print(relations)

            if triplet['Relation'] in relations:
                correctness = 'correct'
                break

        if correctness == '':
            correctness = 'incorrect'

    return correctness


print(check_fact(prompt, extracted_answer, correctness))
