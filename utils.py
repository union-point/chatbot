from langchain_community.document_loaders import  UnstructuredURLLoader,AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer,BeautifulSoupTransformer
from langchain_community.vectorstores import FAISS
from config import configs
from embedings import embeddings
import requests

from langchain.docstore.document import Document


def load_and_transform_html(url : str) -> list[str]:
    """
    Load HTML document from the given URL and transform it.

    Args:
        url (str): The URL to load.

    Returns:
       Sequence[Document]: The transformed HTML document.
    """
    print(f"==>> urls: {url}")

    # Load HTML
    
    loader = AsyncChromiumLoader(url)
    docs = loader.load()
    print(f"==>> docs: {docs}")
    
    response = requests.get(url)
    response.raise_for_status() 
    doc =  Document(page_content=response.text, metadata={"source": url})
    
    html2text = Html2TextTransformer()
    doc_ = html2text.transform_documents(doc)
    
    #bs_transformer = BeautifulSoupTransformer()
    #docs_transformed = bs_transformer.transform_documents(docs, 
    #    tags_to_extract=["h1", "h2", "h3","div",'a','p', "span"])

# Raise an exception if the request was unsuccessful
    print(doc_)
    return doc_


def add_to_db(documents: list[str]) -> None:
    """
    Adds the given documents to the local FAISS index.

    Args:
        documents (List[Document]): The documents to add to the index.
    """
    vectorindex = FAISS.load_local(
        index_name= configs['db_name'], 
        folder_path=configs['db_path'],
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )
    
    FAISS.add_documents(vectorindex, documents=documents)

    vectorindex.save_local(folder_path=configs['db_path'],
                           index_name=configs['db_name'])


def get_k_similar(text: str, k: int) -> list[str]:
    """
    Retrieves the k most similar documents to the given text using a local FAISS index.

    Args:
        text (str): The text for which to find similar documents.
        k (int): The number of similar documents to retrieve.

    Returns:
        List[Document]: A list of the k most similar documents.
    """

    vectorindex = FAISS.load_local(index_name= configs['db_name'],
                                   folder_path=configs['db_path'],
                                   embeddings=embeddings,
                                   allow_dangerous_deserialization=True)
    docs = vectorindex.similarity_search(text, k=k)
    
    return docs

