import json
import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph  # Adjust this import if needed
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)

def load_neo4j_graph(env_path: str = '.env') -> Neo4jGraph:
    # Load from environment
    load_dotenv(env_path, override=True)
    # Optional: OpenAI config if needed elsewhere
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_ENDPOINT = os.getenv('OPENAI_BASE_URL') + '/embeddings' if os.getenv('OPENAI_BASE_URL') else None
    # Initialize Neo4j graph object
    graph = Neo4jGraph(
        url=os.getenv('NEO4J_URI'),
        username=os.getenv('NEO4J_USERNAME'),
        password=os.getenv('NEO4J_PASSWORD'),
        database=os.getenv('NEO4J_DATABASE') or 'neo4j'
    )
    return graph, OPENAI_API_KEY, OPENAI_ENDPOINT


def split_data_from_file(file,name):
    chunks_with_metadata = [] 
    
    #### Load json file
    with open(file, 'r', encoding='utf-8') as f:
        file_as_object = json.load(f)
        keys = list(file_as_object.keys())
    #### pull these keys from the json file
    for subhead in keys: 
        print(f'Processing {subhead} from {file}') 
        #### grab the text of the item
        data = file_as_object[subhead] 
        #### split the text into chunks
        item_text_chunks = text_splitter.split_text(data) 
        chunk_seq_id = 0
        #### loop thtough chunks
        for chunk in item_text_chunks: 
            #### create a record with metadata and the chunk text
            chunks_with_metadata.append({
                'text': chunk, 
                'Source': subhead,
                'chunkSeqId': chunk_seq_id,
                'chunkId': f'{name}-{subhead}-chunk{chunk_seq_id:04d}',
            })
            chunk_seq_id += 1
        print(f'\tSplit into {chunk_seq_id} chunks')
    return chunks_with_metadata