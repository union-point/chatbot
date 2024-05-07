
from langchain_openai import  ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import openai 
import json
from utils import get_k_similar, add_to_db, load_and_transform_html
from config import configs
from typing import Optional


openai.api_key = configs["OPENAI_API_KEY"]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked "
            "to extract, return null for the attribute's value.",
        ), 
        ("human", "{text}"),
    ]
)


class VC_Firm(BaseModel):
    """Information about a Venture Capital Firm."""

    name: Optional[str] = Field(..., description="The name of the Ventenue Capital Firm")
    contacts: Optional[str] = Field(
        ..., description="The contacts of the Ventenue Capital Firm"
    )
    industries: Optional[str] = Field(..., description="industries which they invested in")
    investment_rounds: Optional[str] = Field(..., description="Investment rounds that they participate/lead")


def scrapper(text: str) -> str:
    """
    Use LLM to extract relevant information from a webpage's text.

    Args:
        text (str): The text content of a webpage.

    Returns:
        str: The extracted text content with relevant information.
    """
    
    if not len(text):
        return 'Data not founded'
    print("Extracting content with LLM")


    llm = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        temperature=0,
    )
    runnable = prompt | llm.with_structured_output(
        schema=VC_Firm,
        method="function_calling",
        include_raw=False,
    )
    extracted_content = runnable.invoke({"text": text})
    extracted_dict = VC_Firm.parse_obj(extracted_content).dict()
 
    return json.dumps(extracted_dict)



def get_Chat_response(url):
    #get the content from the webpage
    splits = load_and_transform_html(url,loader="unstructured")
    text = splits[0].page_content
    #scrap the content
    data = scrapper(text)
    #get the similars
    similar_docs = get_k_similar(text = text,k=3)
    list_of_links = [similar_docs[i].metadata["source"] for i in range(len(similar_docs))]
    similars_str = ' '.join(list_of_links)
    #add content to the db
    if url not in list_of_links:
        print('metadata',splits[0].metadata["source"])
        add_to_db(documents=splits)
    
    return f'{data}\n\n{similars_str}'


if __name__ == '__main__':
    print(get_Chat_response("https://www.accel.com/"))
