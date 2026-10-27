import spacy
nlp = spacy.load("en_core_sci_md")
print(f"Model loaded successfully: {nlp.meta['name']} (version {nlp.meta['version']})")
