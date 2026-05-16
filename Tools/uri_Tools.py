import re

def replace_with_generic_uri(query: str) -> str:
    """
    Replace all <...> URIs in a SPARQL query with a generic <uri> placeholder.
    """
    return re.sub(r"<[^>]*>", "<{uri}>", query)

if __name__ == "__main__":
  # Example usage
  query = """
  SELECT DISTINCT ?uri 
  WHERE { 
    <http://dbpedia.org/resource/The_Ultimate_Fighter:_Team_Rousey_vs._Team_Tate> 
    <http://dbpedia.org/property/city> ?uri 
  }
  """

  print(replace_with_generic_uri(query))
