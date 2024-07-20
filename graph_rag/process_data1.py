from llama_index.readers.wikipedia import WikipediaReader

# Initialize WikipediaReader
reader = WikipediaReader()

# Load data from Wikipedia
documents = reader.load_data(pages=['Guardians of the Galaxy Vol. 3'], auto_suggest=False)
