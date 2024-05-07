from langchain_community.document_loaders import  UnstructuredURLLoader,AsyncChromiumLoader,SeleniumURLLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.vectorstores import FAISS
from config import configs
from embedings import embeddings

def load_and_transform_html(url : str, loader:str = "chromium") -> list[str]:
    """
    Load HTML document from the given URL and transform it.

    Args:
        url (str): The URL to load.
        loader (str): The loader to use.("chromium", "selenium", "unstructured")

    Returns:
       Sequence[Document]: The transformed HTML document.
    """
    print(f"==>> urls: {url}")

    # Load HTML
    match loader:
        case "chromium":
            loader = AsyncChromiumLoader([url])
            docs = loader.load()
            bs_transformer = BeautifulSoupTransformer()
            docs = bs_transformer.transform_documents(docs, 
            tags_to_extract=["h1", "h2", "h3","div",'a','p', "span"])
        case "selenium":
            loader = SeleniumURLLoader(urls=[url]) 
            docs = loader.load()
        case "unstructured":
            loader = UnstructuredURLLoader(urls=[url])
            docs = loader.load()
        case _:
            return "error loader is invalid"
    
    print(f'==>> docs size: {len(docs[0].page_content)}')
    return docs


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

