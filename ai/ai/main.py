import ollama 

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
        if len(buffer):
            paragraphs.append((' ').join(buffer))
        return paragraphs
        

def main():
    filename = 'ai/data/cnr.txt'
    paragraphs = parse_file(filename)
    print(paragraphs[:20]) 
    