import ollama
import os 
import json

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

def main():
    filename = 'cnr.txt'
    paragraphs = parse_file(filename)
    embeddings = get_embeddings(filename, 'mistral-openorca', paragraphs)
    print(paragraphs[:20]) # print first 20 paragraphs

