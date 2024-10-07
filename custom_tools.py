import requests
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

# Tool to get the different types of pokemon
@tool
def get_pokemon_types():
    """Fetch Pokemon types"""
    # api url
    url = "https://api.pokemontcg.io/v2/types"
    payload = {}
    headers = {}
    # call api
    response = requests.request("GET", url, headers=headers, data=payload)

    return response

@tool
def search_duckduckgo(query: str) -> str:
    """ Run internet search"""
    search = DuckDuckGoSearchResults()
    results = search.run(query)
    print(f"There was a duck duck go search for {query}")
    return results
