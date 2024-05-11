import time
from pprint import pprint
from nltk.tokenize import word_tokenize

from entity_linking.person_entity_linker import PersonEntityLinker
from entity_linking.organization_entity_linker import OrganizationEntityLinker
from entity_linking.location_entity_linking import LocationEntityLinker

class MultiEntityLinker:

    def __init__(self, person_properties, organization_properties, location_properties, stopwords):

        self.person_properties = person_properties
        self.organization_properties = organization_properties
        self.location_properties = location_properties

        self.stopwords = stopwords

        self.person_linker = PersonEntityLinker(self.person_properties)
        self.organization_linker = OrganizationEntityLinker(self.organization_properties)
        self.location_linker = LocationEntityLinker(self.location_properties)

    def preprocess_entity_name(self, entity_label: str) -> str:

        title_tokens = word_tokenize(entity_label)

        if title_tokens[0].lower() in self.stopwords:
            return ' '.join(title_tokens[1:])
        else:
            return entity_label
    
    def extract_entities(self, entity_dict: dict, text: str, num_search_results: int, context_window: int, lang_code: str):

        for entity, information in entity_dict.items():

            start_char, entity_type = information

            start = time.time()

            processed_entity = self.preprocess_entity_name(entity)

            if processed_entity != '':

                print("="*120)
                print(f"Original label: {entity}")
                print()
                print(f"Attempting to match {entity_type} entity: {processed_entity}")
                print()

                time.sleep(1.5)

                if entity_type.lower() == "per":
                    entity_info = self.person_linker.person_entity_extraction(text=text, 
                                                                            entity=processed_entity, 
                                                                            start_char=start_char, 
                                                                            num_search_results=num_search_results, 
                                                                            context_window=context_window,
                                                                            lang_code=lang_code)
                elif entity_type.lower() == "org":
                    entity_info = self.organization_linker.organization_entity_extraction(text=text, 
                                                                            entity=processed_entity, 
                                                                            start_char=start_char, 
                                                                            num_search_results=num_search_results, 
                                                                            context_window=context_window,
                                                                            lang_code=lang_code)
                elif entity_type.lower() == "loc" or entity_type.lower() == "gpe":
                    entity_info = self.location_linker.location_entity_extraction(text=text, 
                                                                            entity=processed_entity, 
                                                                            start_char=start_char, 
                                                                            num_search_results=num_search_results, 
                                                                            context_window=context_window,
                                                                            lang_code=lang_code)

                end = time.time()

                pprint(entity_info)
                print()
                print(f"Time elapsed: {end-start}")
                print("="*120)
                print()


