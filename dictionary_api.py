import requests

def get_definitions_and_synonyms(word):
    """
    Usa DictionaryAPI.dev para definiciones, sinónimos, ejemplos en inglés.
    """
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    r = requests.get(url)
    if r.status_code != 200:
        return {
            "definitions": ["No definition found."],
            "synonyms": [],
            "examples": []
        }

    data = r.json()
    definitions_list = []
    synonyms_list = set()
    examples_list = []

    if isinstance(data, list) and len(data) > 0:
        for meaning_obj in data[0].get("meanings", []):
            for def_obj in meaning_obj.get("definitions", []):
                definition = def_obj.get("definition", "")
                if definition:
                    definitions_list.append(definition)

                syns = def_obj.get("synonyms", [])
                for syn in syns:
                    synonyms_list.add(syn)

                example = def_obj.get("example", "")
                if example:
                    examples_list.append(example)

    return {
        "definitions": definitions_list if definitions_list else ["No definition found."],
        "synonyms": list(synonyms_list),
        "examples": examples_list
    }
