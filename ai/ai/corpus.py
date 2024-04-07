from pathlib import Path

def fname(): 
    directory = '/Users/saksham/Read-N-RAG/ai/ai/data'
    list = []
    files = Path(directory).glob('*.txt')

    for file in files:
        p_name = str(file).split('/')[-1]
        list.append(p_name)
    
    return list    



