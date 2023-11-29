from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


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
def fetch_entity_id(entity):
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


# Fetch the name and aliases of the relation obtained from Wikidata
def fetch_name_and_aliases_wikidata(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    relations = []

    # Scrape the relationLabel page to get the name of the relation
    try:
        span_element = soup.find("span", class_="wikibase-title-label")
        relation_name = span_element.text
        relations.append(relation_name)
    except:
        pass

    # Scrape the relationLabel page to get all the aliases of the relation
    try:
        ul_tag = soup.find("ul", class_="wikibase-entitytermsview-aliases")
        if ul_tag:
            li_tags = ul_tag.find_all("li")
            for li_tag in li_tags:
                relaton_alias = li_tag.text
                relations.append(relaton_alias)
    except:
        pass

    return relations


# Perform a SPARQL query to get the relation between two entities from Wikidata
def sparql_query_wikidata(subj, obj):
    if len(subj) == 0 or len(obj) == 0:
        relations = ['No relation found']
    else:
        subj_id = fetch_entity_id(subj)
        obj_id = fetch_entity_id(obj)

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
            relations = ['No relation found']
        else:
            relation_label = data['results']['bindings'][0]['relationLabel']['value']
            relations = fetch_name_and_aliases_wikidata(relation_label)

    return relations


# Fetch the name and aliases of the relation obtained from DBpedia
def fetch_name_and_aliases_dbpedia(url):
    # Start the browser driver, configure the relevant parameters to prevent the browser window from popping up
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    # Open the webpage
    driver.get(url)
    # Execute JavaScript code to load dynamic data
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    # Fetch the webpage source code
    html = driver.page_source
    # Close the browser driver
    driver.quit()

    soup = BeautifulSoup(html, 'html.parser')
    relations = []

    # Fetch the name of the relation
    try:
        span_tag = soup.find("span", property="rdfs:label", lang="en")
        relation_name = span_tag.text
        relations.append(relation_name)
    except:
        pass

    # Fetch the aliases of the relation (through the link of equivalent property)
    try:
        a_tags = soup.find_all("a", class_="uri", rel="owl:equivalentProperty")
        for a_tag in a_tags:
            href = a_tag.get("href")
            if "wikidata.org" in href:
                wiki_url = href.replace("http://www.wikidata.org/entity/", "https://www.wikidata.org/wiki/Property:")
                break

        relation_aliases = fetch_name_and_aliases_wikidata(wiki_url)
        for relation_alias in relation_aliases:
            relations.append(relation_alias)
    except:
        pass

    return relations


# Perform a SPARQL query to get the relation between two entities from DBpedia
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
    if len(data['results']['bindings']) == 0:
        relations = ['No relation found']
    else:
        relations = []
        for i in range(len(data['results']['bindings'])):
            r_url = data['results']['bindings'][i]['p']['value']
            if 'wikiPageWikiLink' in r_url:
                continue
            elif 'property' in r_url:
                relation = fetch_name_and_aliases_dbpedia(r_url)
                for item in relation:
                    relations.append(item)
            elif 'ontology' in r_url:
                relation = fetch_name_and_aliases_dbpedia(r_url)
                for item in relation:
                    relations.append(item)

    return relations


# print(sparql_query_wikidata("Nicaragua", "Managua"))
# print(sparql_query_wikidata('Barack Obama', 'Hawaii'))
# print(sparql_query_dbpedia('Nicaragua', 'Managua'))
# print(sparql_query_dbpedia('Barack Obama', 'Hawaii'))

# prompt = 'Is Beijing the capital of China?'
# prompt = "Paris is capital of Nicaragua"
# prompt = 'What is the capital of Nicaragua?'
# prompt = "The capital of China is ..."
prompt = "Barack Obama was born in Hawaii. Yes or no?"
extracted_answer = 'yes'
# extracted_answer = 'Beijing'
correctness = ''

triplet_extractor = pipeline(
    'text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large'
)


def check_fact(prompt, extracted_answer, correctness):
    # We need to use the tokenizer manually since we need special tokens
    extracted_text = triplet_extractor.tokenizer.batch_decode(
        [triplet_extractor(prompt, return_tensors=True, return_text=False)[0]["generated_token_ids"]]
    )

    extracted_triplets = extract_triplets(extracted_text[0])
    print(extracted_triplets)

    if extracted_answer in ['yes', 'no']:
        for triplet in extracted_triplets:
            relations = []
            relations_list = []
            relations_list.append(sparql_query_wikidata(triplet['Subject'], triplet['Object']))
            relations_list.append(sparql_query_dbpedia(triplet['Subject'], triplet['Object']))
            print(relations_list)
            for item in relations_list:
                for relation in item:
                    relations.append(relation)
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

        # Check if the prompt contains negation
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
            relations_list = []
            relations_list.append(sparql_query_wikidata(triplet['Subject'], extracted_answer))
            relations_list.append(sparql_query_wikidata(extracted_answer, triplet['Object']))
            relations_list.append(sparql_query_dbpedia(triplet['Subject'], extracted_answer))
            relations_list.append(sparql_query_dbpedia(extracted_answer, triplet['Object']))
            print(relations_list)
            for item in relations_list:
                for relation in item:
                    relations.append(relation)
            print(relations)

            if triplet['Relation'] in relations:
                correctness = 'correct'
                break

        if correctness == '':
            correctness = 'incorrect'

    return correctness


print(check_fact(prompt, extracted_answer, correctness))
