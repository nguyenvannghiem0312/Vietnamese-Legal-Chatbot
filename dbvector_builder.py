import argparse
import sys
from pathlib import Path
from typing import List

from embedding.embedder import EmbedderHuggingFace
from embedding.vector_store import VectorMemory

from utils.log import get_logger
from typing import List, Any
import os

import glob
import json
from typing import List, Dict, Any
from datasets import load_dataset
from pyvi.ViTokenizer import tokenize
from tqdm import tqdm

from llama_index.core.node_parser import SentenceSplitter

logger = get_logger(__name__)

class Document:
    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}

def load_documents(docs_path: str, file_type: str = 'md', vitokenizer: bool = True) -> List[Document]:
    """
    Loads documents from the specified path based on the file type.

    Args:
        docs_path (str): The path to the documents.
        file_type (str): The type of documents to load. Options are 'md', 'txt', 'json', 'huggingface'.

    Returns:
        List[Document]: A list of loaded documents.

    Raises:
        ValueError: If an unsupported file type is provided.

    Example:
        >>> load_documents('/path/to/documents', 'md')
        [document1, document2, document3]
    """

    # Kiểm tra sự tồn tại của đường dẫn
    if not os.path.exists(docs_path) and file_type != 'huggingface':
        logger.error(f"The specified path does not exist: {docs_path}")
        raise FileNotFoundError(f"The specified path does not exist: {docs_path}")
    
    documents = []
    if file_type == 'md':
        for filepath in glob.glob(os.path.join(docs_path, '**/*.md'), recursive=True):
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(Document(content=content, metadata={"source": filepath}))
    elif file_type == 'txt':
        for filepath in glob.glob(os.path.join(docs_path, '**/*.txt'), recursive=True):
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(Document(content=content, metadata={"source": filepath}))
    elif file_type == 'json':
        for filepath in glob.glob(os.path.join(docs_path, '**/*.json'), recursive=True):
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Assuming JSON files contain a list of documents
                if isinstance(data, list):
                    for item in data:
                        content = json.dumps(item)
                        documents.append(Document(content=content['corpus'], metadata={"source": filepath, "id": content['corpus_id']}))
                else:
                    content = json.dumps(data)
                    documents.append(Document(content=content, metadata={"source": filepath}))
    elif file_type == 'huggingface':
        dataset = load_dataset(docs_path)['corpus'].to_list()
        for i in tqdm(range(len(dataset))):
            if vitokenizer == True:
                content = tokenize(str(dataset[i]['corpus']))
            else:
                content = str(dataset[i]['corpus'])
            documents.append(Document(content=content, metadata={"source": docs_path, "id": dataset[i]['corpus_id']}))
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return documents

def split_chunks(sources: List[Document], chunk_size: int = 512, chunk_overlap: int = 0) -> List:
    """
    Splits a list of sources into smaller chunks.

    Args:
        sources (List): The list of sources to be split into chunks.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 512.
        chunk_overlap (int, optional): The amount of overlap between consecutive chunks. Defaults to 0.

    Returns:
        List: A list of smaller chunks obtained from the input sources.
    """
    chunks = []

    text_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=' ')
    for doc in tqdm(sources, desc="Split corpus"):
        cur_text_chunks = text_parser.split_text(doc.content)
        chunks.extend([Document(content=cur_text_chunk, metadata=doc.metadata) for cur_text_chunk in cur_text_chunks])
    return chunks

def build_memory_index(docs_path: str, file_type: str, vector_store_path: str, chunk_size: int, chunk_overlap: int, vitokenizer: bool):
    sources = load_documents(str(docs_path), file_type=file_type, vitokenizer=vitokenizer)
    logger.info(f"Number of Documents: {len(sources)}")
    chunks = split_chunks(sources, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info(f"Number of Chunks: {len(chunks)}")
    embedding = EmbedderHuggingFace().get_embedding()
    VectorMemory.create_memory_index(embedding, chunks, vector_store_path)
    logger.info("Memory Index has been created successfully!")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory Builder")
    parser.add_argument(
        "--docs-path",
        type=str,
        help="The path of documents.",
        required=False,
        default='docs/text',
    )
    parser.add_argument(
        "--file-type",
        type=str,
        help="The file type of documents. Can be 'txt', 'md', 'json', 'huggingface'",
        required=False,
        default='txt',
    )
    parser.add_argument(
        "--vitokenizer",
        type=bool,
        help="Use the Pyvi to Vitokenizer",
        required=False,
        default=True,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="The maximum size of each chunk. Defaults to 512.",
        required=False,
        default=512,
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        help="The amount of overlap between consecutive chunks. Defaults to 0.",
        required=False,
        default=0,
    )
    parser.add_argument(
        "--vector-store-path",
        type=str,
        help="The path of vector store.",
        required=False,
        default='vectordb/chroma',
    )

    return parser.parse_args()

def main(parameters):
    root_folder = Path(__file__).resolve().parent.parent
    doc_path = root_folder / "docs"
    vector_store_path = root_folder / "vector_store" / "docs_index"

    build_memory_index(
        docs_path = parameters.docs_path,
        file_type = parameters.file_type,
        chunk_size = parameters.chunk_size,
        chunk_overlap = parameters.chunk_overlap,
        vector_store_path = parameters.vector_store_path,
        vitokenizer= parameters.vitokenizer,
    )


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)