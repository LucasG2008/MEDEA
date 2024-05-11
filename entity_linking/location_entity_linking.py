import requests
import numpy as np
from entity_linking.entity_linking import EntityLinker
from sentence_transformers import SentenceTransformer

class LocationEntityLinker(EntityLinker):

    def __init__(self, location_properties: dict):

        super().__init__()

        self.location_properties = location_properties

        # official language, population, contains administrative territorial entity, GeoNames ID
        self.filter_properties = ["P37", "P1082", "P150", "P1566"]

        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def get_location_qnumber(self, wikiarticle: str, wikisite: str, lang_code: str) -> str:
        """
        Make sure Q-ID from wikidata API corresponds to a location

        Input:
        - wikiarticle: Exact name of wikidata article
        - wikisite: Language specific wikidata site to make API call to
        - lang_code: Wikidata language code

        Output: 
        - qid: Wikidata Q-ID belonging to a location
        """
        qid = self.get_qnumber(wikiarticle, wikisite)

        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        response = requests.get(url)

        if response.status_code == 200:
            try:
                data = response.json()
                entity_type = data["entities"][qid]["type"]
                
                # Checking official language, population and contains administrative territorial entity
                if entity_type == "item":
                    if any(loc_prop in data["entities"][qid]["claims"] for loc_prop in self.filter_properties):
                        return qid
            except KeyError:
                print(f"Error: QID {qid} does not exist or does not have type information.")
            except Exception as e:
                print(f"Error: {e}")

        return '-1'
    
    def filter_location_qids(self, qid_list: list) -> list:
        """
        Filter list of Q-IDs to only include people entities

        Input:
        - qid_list: List of Q-IDs to filter

        Output: 
        - people_qids: List of only people Q-IDs
        """
        location_qids = []

        for qid in qid_list:
            url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
            response = requests.get(url)

            if response.status_code == 200:
                try:
                    data = response.json()
                    entity_type = data["entities"][qid]["type"]
                    
                    # Checking coordinate location, population and area
                    # Checking official language, population and contains administrative territorial entity
                    if entity_type == "item":
                        if any(loc_prop in data["entities"][qid]["claims"] for loc_prop in self.filter_properties):
                            location_qids.append(qid)
                except KeyError:
                    print(f"Error: QID {qid} does not exist or does not have type information.")
                except Exception as e:
                    print(f"Error: {e}")

        print("QID's Before Filtering: ", len(qid_list))
        print("QID's After Filtering: ", len(location_qids))

        return location_qids
    
    def location_wikidata_search(self, search_phrase: str, num_results: int, lang_code: str):
        """
        Make API search request to wikidata API

        Input:
        - search_phrase: Phrase to search wikidata API with
        - num_results: Number of results to return for wikidata API search
        - lang_code: Wikidata language code

        Output: 
        - results_information: Dictionary containing Q-IDs and corresponding text snippets
        """
        resp = requests.get('https://www.wikidata.org/w/api.php', {
            "action": "query",
            "format": "json",
            "uselang": lang_code,
            "list": "search",
            "formatversion": "2",
            "srsearch": search_phrase,
            "srnamespace": "0",
            "srlimit": "max",
            "srinfo": "totalhits|suggestion|rewrittenquery",
            "srprop": "snippet|extensiondata",
            "srenablerewrites": 1,
            "srsort": "relevance"
        }).json()

        search_results = resp['query']['search']
        search_results = search_results[:num_results]

        results_information = {result['title']:result['snippet'] for result in search_results if result['snippet'] != ''}
        results_qids = list(results_information.keys())

        filtered_qids = self.filter_location_qids(results_qids)

        weights = np.linspace(1, 0.01, num_results)
        final_results = {qid:[snippet, weights[results_qids.index(qid)]] for qid, snippet in results_information.items() if qid in filtered_qids}

        print("Candidates: ", list(final_results.values()))
        print()

        return final_results

    def location_entity_extraction(self, text: str, entity: str, start_char: int, num_search_results: int, context_window: int, lang_code: str) -> dict:
        """
        Extract relevant entity information from wikidata corpus for a given location entity

        Input:
        - text: Text in which entity is mentioned
        - entity: String representation of entity
        - start_char: Starting character of entity
        - num_search_results: Number of results to return for a search
        - context_window: How much of the original text surrounding an entity to use for entity linking
        - lang_code: Wikidata language code

        Output: 
        - top_entity: Entity with closest relation to context
        """

        if (qid := self.get_location_qnumber(entity, f'{lang_code}wiki', lang_code=lang_code)) != '-1':

            print("Found exact entity match")
            location_info = self.extract_wikidata_entity_info(qid, self.location_properties, lang_code=lang_code)

        
        # Extract entity and relevant information from wikidata corpus
        else:
            search_results = self.location_wikidata_search(entity, num_results=num_search_results, lang_code=lang_code)

            context = self.extract_context_by_words(text, start_char, context_window=context_window)

            if search_results == {}:
                print(f"Entity matching for {entity} failed, search yielded no results")
                return {}

            top_entity = self.context_entity_matching(context, search_results, self.model)

            location_info = self.extract_wikidata_entity_info(top_entity, self.location_properties, lang_code=lang_code)

        return location_info