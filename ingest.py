import argparse
import hashlib
from pathlib import Path
import fitz
from app.knowledge import embed_documents, get_index, embed_query


def chunks(text:str, chunk_size:int=1100, overlap:int=200):
    """
    Splits the input text into chunks of specified size with a given overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk. Default is 1100 characters.
        overlap (int): The number of overlapping characters between consecutive chunks. Default is 180 characters.

    Returns:
        List[str]: A list of text chunks.
    """
    text = ' '.join(text.split())  # Normalize whitespace
    step = chunk_size - overlap
    return [text[i:i + chunk_size] for i in range(0, len(text), step) if len(text[i:i + chunk_size]) > 80]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', default='Data/introduction_merged.pdf')
    args = parser.parse_args()
    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f'PDF not found: {pdf}')
    index = get_index()
    with fitz.open(pdf) as doc:
        batch = []
        for page_number, page in enumerate(doc, start=1):
            for part, content in enumerate(chunks(page.get_text()), start=1):
                key = hashlib.sha256(f'{pdf.name}:{page_number}:{part}:{content}'.encode()).hexdigest()
                batch.append((key,content,{'page':page_number,'text':content,'file':pdf.name}))
                if len(batch) >= 50:
                    vectors = embed_documents([b[1] for b in batch])
                    index.upsert(vectors=[{'id':b[0],'values':v,'metadata':b[2]} for b,v in zip(batch,vectors)])
                    batch=[]
            if page_number % 50 == 0:
                print(f'Indexed pages: {page_number}/{len(doc)}',flush=True)
        if batch:
            vectors=embed_documents([b[1] for b in batch])
            index.upsert(vectors=[{'id':b[0],'values':v,'metadata':b[2]} for b,v in zip(batch,vectors)])
    print('Ingestion complete')



if __name__ == '__main__': 
    main()
  