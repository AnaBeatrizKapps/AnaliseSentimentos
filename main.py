import spacy
text = """
Esse é um ótimo teste de análise de sentimentos
"""

nlp = spacy.load("pt_core_news_sm")
doc = nlp(text)

#Tokening
token_list = [token for token in doc]

#Removing Stop Words
filtered_tokens = [token for token in doc if not token.is_stop]

lemmas = [f"Token: {token}, lemma: {token.lemma_}" for token in filtered_tokens ]

#Vectoring Text
print(filtered_tokens[1].vector)