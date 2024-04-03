import ollama
import os 
import json
from numpy.linalg import norm
import numpy as np

def parse_file(filename):
    with open(filename, encoding='utf-8-sig') as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((' ').join(buffer))
                buffer = []
        if len(buffer): # if buffer is not empty after exiting loop append it to paragraphs
            paragraphs.append((' ').join(buffer))
        return paragraphs

def load_embeddings(filename):
    if not os.path.exists(f'embeddings/{filename}.json'):
        return False
    #loading the embeddings from json
    with open(f'embeddings/{filename}.json', 'r') as f:
        return json.load(f)
        
def get_embeddings(filename, modelname, chunks):
    # checking for already saved embeddings
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    
    embeddings = [
        ollama.embeddings(model = modelname, prompt=chunk)['embedding']
        for chunk in chunks
    ]
    # save the embeddings to json
    save_embeddings(filename, embeddings)
    return embeddings

def save_embeddings(filename, embeddings):
    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')
    
    with open(f'embeddings/{filename}.json', 'w') as f:
        json.dump(embeddings, f)

def consine_sim(needle, haystack):   
    # calculate the norm of the needle
    needleNorm = norm(needle)
    
    # calculate the cosine similarity between the needle and each paragraph in the haystack
    sim_scores = [ np.dot(needle,item) / (needleNorm * norm(item)) for item in haystack ]
    
    # return the sorted list of tuples containing the similarity score and the index of the paragraph
    return sorted(zip(sim_scores, range(len(haystack))), reverse = True)


def main():
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """
    filename = 'cnr.txt'
    paragraphs = parse_file(filename)
    embeddings = get_embeddings(filename, 'mistral-openorca', paragraphs)
    
    prompt = input('What would you like to know about the text? ')
    promt_embedding = ollama.embeddings(model = 'mistral-openorca', prompt=prompt)['embedding']
    
    most_sim_chunks = consine_sim(promt_embedding, embeddings)[:20]
    
    #iterating through the most similar paragraphs
    # for item in most_sim_chunks:
    #     print(f"Similarity score: {item[0]} ", paragraphs[item[1]], '\n')
        
    response = ollama.chat(
        model= 'mistral-openorca',
        messages= [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT + '\n'.join([paragraphs[item[1]] for item in most_sim_chunks]),
            },
            {
                'role': 'user',
                'content': prompt
            },
        ],
    )
    print('\n')
    print(response['message']['content'])
    
