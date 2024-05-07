
from langchain_community.vectorstores import FAISS

from config import configs
from utils import load_and_transform_html

from embedings import embeddings


if __name__ == '__main__':
    urls = [
    'https://www.accel.com',
    'https://www.a16z.com',
    'https://www.greylock.com',
    'https://www.benchmark.com',
    'https://www.sequoiacap.com',
    'https://www.indexventures.com',
    'https://www.kpcb.com',
    'https://www.lsvp.com',
    'https://www.matrixpartners.com',
    'https://www.500.co',
    'https://www.sparkcapital.com',
    'https://www.insightpartners.com',
    ]


    # Pass the documents and embeddings inorder to create FAISS vector index
    docs = load_and_transform_html(urls[0])
    vectorindex= FAISS.from_documents(docs, embeddings)

    for i in range(1,len(urls)):
        docs = load_and_transform_html(urls[i])
        FAISS.add_documents(vectorindex,documents=docs)
        
    # save to disk
    vectorindex.save_local(folder_path=configs['db_path'],index_name=configs['db_name'])

