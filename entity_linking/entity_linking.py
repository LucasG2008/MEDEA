# Imports
import re
import requests
import numpy as np

import time

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

from SPARQLWrapper import SPARQLWrapper, JSON

class EntityLinker():

    def __init__(self):
        pass

    def get_qnumber(self, wikiarticle: str, wikisite: str) -> str:
        """
        Get Wiki data Q-ID from site name using the wikidata API

        Input:
        - wikiarticle: Exact name of wikidata article
        - wikisite: Language specific wikidata site to make API call to

        Output: 
        - qid: Wikidata Q-ID
        """

        resp = requests.get('https://www.wikidata.org/w/api.php', {
            'action': 'wbgetentities',
            'titles': wikiarticle,
            'sites': wikisite,
            'props': '',
            'format': 'json'
        }).json()

        q_number = list(resp['entities'])[0]

        # Check if QID corresponds to an disambiguation page
        if self.check_wikimedia_disambiguation_page(q_number, lang_code="en") == True:
            return '-1'

        return q_number
    
    def check_wikimedia_disambiguation_page(self, qid: str, lang_code: str) -> bool:
        """
        Check if wikidata Q-ID corresponds to a wikimedia disambiguation page

        Input:
        - qid: Q-ID for wikidata article
        - lang_code: Wikidata language code

        Output: 
        - is_disamiguation_page: Boolean determining if Q-ID corresponds to wikidata disambiguation page
        """
        # Define the Wikidata API endpoint URL
        url = "https://www.wikidata.org/w/api.php"

        # Define the parameters for the entity query
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": qid,
            "languages": lang_code
        }

        # Send the request and get the response
        response = requests.get(url, params=params)
        data = response.json()

        # Check if the entity exists
        if "entities" in data and qid in data["entities"]:
            # Get the English description of the entity
            description = data["entities"][qid]["descriptions"][lang_code]["value"]
            if description == 'Wikimedia disambiguation page':
                return True
        else:
            return False
    
    def get_entity_info(self, qid: str) -> dict:
        """
        Extract all data from wikidata API using Q-ID

        Input:
        - qid: Q-ID key for wikidata page

        Output: 
        - data: Dictionary containing wikidata information
        """
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return data["entities"][qid]
        else:
            print("Error: Unable to fetch data from Wikidata API")
            return None
        
    def get_entity_property_values(self, qid, property_id, lang_code):

        endpoint = SPARQLWrapper("https://query.wikidata.org/sparql")
    
        query = """
        SELECT ?value ?valueLabel
        WHERE {{
        wd:{0} p:{1} ?statement.
        ?statement ps:{1} ?value.
        OPTIONAL {{ ?value rdfs:label ?valueLabel. FILTER(LANG(?valueLabel) = "{2}") }}
        }}
        """.format(qid, property_id, lang_code)
        
        # Set the SPARQL query and return format
        endpoint.setQuery(query)
        endpoint.setReturnFormat(JSON)
        
        try:
            result = endpoint.query().convert()
            
            values = []
            for binding in result["results"]["bindings"]:
                if "value" in binding:
                    # Handle date properties separately
                    if binding["value"]["type"] == "literal" and binding["value"]["datatype"] == "http://www.w3.org/2001/XMLSchema#dateTime":
                        values.append(binding["value"]["value"])
                    else:
                        values.append(binding["valueLabel"]["value"] if "valueLabel" in binding else binding["value"]["value"])
            
            return values
        
        except Exception as e:
            print("An error occurred:", str(e))
            return None
    
    def extract_wikidata_entity_info(self, wikidata_qid: str, entity_properties: dict, lang_code: str) -> dict:
        """
        Extract relevant information about an entity from wikidata API

        Input:
        - wikidata_qid: Q-ID key for wikidata page
        - person_property_info: Dictionary containing relevant wikidata property keys and labels
        - language_code: language for data extraction

        Output: 
        - person_info: Dictionary containing wikidata information
        """
        # Make API call to wikidata with persons qid
        item_info = self.get_entity_info(wikidata_qid)

        entity_info = {}

        # Extract relevant information from the wikidata API
        if item_info:

            # Check if the entity name is available in the selected language, otherwise use english
            if lang_code in list(item_info["labels"].keys()):
                entity_label = item_info["labels"][lang_code]["value"]
            else:
                entity_label = item_info["labels"]["en"]["value"]

            entity_info['label'] = entity_label

            # Check if the description is available in the selected language, otherwise use english
            if lang_code in list(item_info["descriptions"].keys()):
                description = item_info["descriptions"][lang_code]["value"]
            else:
                description = item_info["descriptions"]["en"]["value"]

            entity_info['description'] = description
            
            for prop in item_info["claims"]:
                if prop in entity_properties.keys():
                    property_data = []
                    for claim in item_info["claims"][prop]:
                        if claim["mainsnak"]['snaktype'] == 'value':
                            value = claim["mainsnak"]["datavalue"]["value"]
                            if "id" in value:
                                try:
                                    value_url = f"https://www.wikidata.org/wiki/Special:EntityData/{value['id']}.json"
                                    value_response = requests.get(value_url)
                                    if value_response.status_code == 200:
                                        value_data = value_response.json()

            #                            # Check if the property data is available in the selected language, otherwise use english
                                        if lang_code in value_data["entities"][value["id"]]["labels"].keys():
                                            label = value_data["entities"][value["id"]]["labels"][lang_code]["value"]
                                        else:
                                            label = value_data["entities"][value["id"]]["labels"]["en"]["value"]
                                        property_data.append(label)
                                except:
                                    continue
                            else:
                                property_data.append(value)
                    if len(property_data) == 0:
                        entity_info[entity_properties[prop]] = property_data
                    else:
                        entity_info[entity_properties[prop]] = property_data if len(property_data) > 1 else property_data[0]

            # SPEEDIER BUT WORSE METHOD

            #for prop in item_info["claims"]:
            #    if prop in entity_properties.keys():
            #        property_values = self.get_entity_property_values(wikidata_qid, prop, lang_code)
            #        entity_info[entity_properties[prop]] = property_values

        return entity_info
    
    def extract_context_by_sentences(self, text: str, start_char: int, context_window: int = 0) -> str:
        """
        Extract relevant context sentences/words for a given entity

        Input:
        - text: Text containing given entity
        - start_char: Starting character number in text for given entity
        - context_window: Number of sentences before and after the given entity's sentence to include in context

        Output: 
        - context: String containing context for a given entity in a text
        """
        character_count = 0

        sentences = sent_tokenize(text)
        for sentence in sentences:
            character_count += len(sentence)
            if character_count > start_char:
                selected_sentence_id = sentences.index(sentence)
                break

        if context_window == 0:
            return sentences[selected_sentence_id]
        
        minimum_id = selected_sentence_id - context_window
        maximum_id = selected_sentence_id + context_window + 1
        
        context_sentence_ids = list(np.arange(minimum_id, maximum_id))

        clip_function = lambda x: min(max(x, 0), len(sentences)-1)
        clipped_context_sentence_ids = set(map(clip_function, context_sentence_ids))

        context = ''.join(sentences[context_id] for context_id in clipped_context_sentence_ids)

        return context
    
    def extract_context_by_words(self, text: str, start_char: int, context_window: int = 0) -> str:
        """
        Extract relevant context sentences/words for a given entity

        Input:
        - text: Text containing given entity
        - start_char: Starting character number in text for given entity
        - context_window: Number of words before and after the given entity to include in context

        Output: 
        - context: String containing context for a given entity in a text
        """
        character_count = 0

        #words = word_tokenize(text)
        words = re.split('\s', text)

        #if start_char > ''.join(sentences[:-1])
        for word in words:
            character_count += len(word) + 1
            if character_count > start_char:
                selected_word_id = words.index(word)
                break

        if context_window == 0:
            return words[selected_word_id]
        
        minimum_id = selected_word_id - context_window
        maximum_id = selected_word_id + context_window + 1
        
        context_word_ids = list(np.arange(minimum_id, maximum_id))

        clip_function = lambda x: min(max(x, 0), len(words)-1)
        clipped_context_word_ids = set(map(clip_function, context_word_ids))
        clipped_context_word_ids = sorted(clipped_context_word_ids)

        context = ' '.join(words[context_id] for context_id in clipped_context_word_ids)

        return context
    
    def context_entity_matching(self, context: str, entity_candidates: dict, sentence_transformer: SentenceTransformer) -> str:
        """
        Use text embeddings to calculate most similar entry in wikidata corpus to given entity using context

        Input:
        - context: Context surrounding entity in text
        - entity_candidates: Dictionary of possible entities from wikidata corpus
        - sentence_transformer: Text embedding model to vectorize text

        Output: 
        - top_entity: Entity with closest relation to context
        """
        entity_information = list(entity_candidates.values())
        entity_names = list(entity_candidates.keys())

        entity_snippets = [info[0] for info in entity_information]
        entity_weights = [info[1] for info in entity_information]

        entity_snippets_embeddings = sentence_transformer.encode(entity_snippets, convert_to_tensor=True)

        context_embedding = sentence_transformer.encode(context, convert_to_tensor=True)

        # TODO Implement similarity threshold for entity disambiguation

        cos_scores = util.cos_sim(context_embedding, entity_snippets_embeddings)[0]
        cos_scores = cos_scores.cpu()
        cos_scores = np.array(cos_scores)

        weighted_scores = cos_scores*np.array(entity_weights)

        max_score = weighted_scores.max()

        top_entity_id = list(weighted_scores).index(max_score)

        top_entity = entity_names[top_entity_id]

        print("Similarity Score: ", max_score)

        return top_entity
    
    def entity_label_matching(self, entity_label: str, candidate_labels: dict, sentence_transformer: SentenceTransformer) -> str:
        """
        Use text embeddings to calculate most similar entry in wikidata corpus to given entity based on their labels

        Input:
        - entity_label: Label/Title for the given entity
        - candidate_labels: Dictionary of entity candiates and their weights
        - sentence_transformer: Text embedding model to vectorize text

        Output: 
        - top_entity: Candidate entity with closest relation to entity
        """

        entity_weights = list(candidate_labels.values())
        entity_names = list(candidate_labels.keys())

        entity_names_embeddings = sentence_transformer.encode(entity_names, convert_to_tensor=True)

        entity_label_embedding = sentence_transformer.encode(entity_label, convert_to_tensor=True)

        # TODO Implement similarity threshold for entity disambiguation

        cos_scores = util.cos_sim(entity_label_embedding, entity_names_embeddings)[0]
        cos_scores = cos_scores.cpu()
        cos_scores = np.array(cos_scores)

        weighted_scores = cos_scores*np.array(entity_weights)

        max_score = weighted_scores.max()

        top_entity_id = list(weighted_scores).index(max_score)

        top_entity = entity_names[top_entity_id]

        print("Similarity Score: ", max_score)

        return top_entity