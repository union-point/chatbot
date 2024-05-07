
from langchain.chains import create_extraction_chain
from langchain_openai import  ChatOpenAI

import openai 

from utils import get_k_similar,add_to_db,load_and_transform_html
from config import configs

openai.api_key = configs["OPENAI_API_KEY"]

llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-0613")

def scrapper(text: str) -> str:
    """
    Use LLM to extract relevant information from a webpage's text.

    Args:
        text (str): The text content of a webpage.

    Returns:
        str: The extracted text content with relevant information.
    """
    schema = {
        "properties": {
            "Company Name": {"type": "string"},
            "contacts": {"type": "array", "items": {}},
            "all Industries that they invest in": {"type": "array", "items": {}},
            "Investment rounds that they participate/lead": {"type": "array", "items": {}}
        },
        "required": ["Company Name", "contacts", "all Industries that they invest in", "Investment rounds that they participate/lead"],
    }

    print("Extracting content with LLM")

    # Process the first split
    extracted_content = create_extraction_chain(schema=schema, llm=llm).invoke(text)["text"]
    return str(extracted_content)





function_extraction=[
        {
            "name": "information_extraction",
            "description": "Parse unstructured data nicely",
            "parameters": {
                'type': 'object',
                'properties': {
                    'name': {
                             'type': 'string'},
                    'contacts': {
                                 'type': 'array', 
                                 'items': {}},
                    'industries_invested_in': {
                                        'type': 'array',
                                        'items': {}
                                    },
                    'investment_rounds': {
                                        'type': 'array',
                                        'items': {}
                                    }
                    },
                    'required': ['name', 'contacts', 'industries_invested_in', 'investment_rounds']
            }
        }
    ]

def custom_scrapper(text: str) -> str:
    """
    Extract and save the relevant entities mentioned in the following passage together with their properties.

    Args:
        text (str): The raw HTML data to be parsed

    Returns:
        str: A JSON string containing the extracted data
    """
    
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo-0613",  # Feel free to change the model to gpt-3.5-turbo-1106
        messages=[
            {"role": "system", "content": "You are a master at scraping and parsing raw HTML."},
            {"role": "user", "content": f'''
                Extract and save the relevant entities mentioned in the following passage together with their properties.

                Only extract the properties mentioned in the 'information_extraction' function.

                If a property is not present and is not required in the function parameters, do not include it in the output.

                Passage:
                {text}
                '''}
        ],
        functions=function_extraction,
        function_call="auto"
    )

    argument_str = completion.choices[0].message.function_call.arguments  # type: str
    data = argument_str
    return data



def get_Chat_response(user_input):
    url = user_input
    splits = load_and_transform_html(url)
    text = splits[0].page_content
    if len(text) == 13:
        
        print(text)
    data = custom_scrapper(text)

    similar_docs = get_k_similar(text = text,k=3)
    
    list_of_links = [similar_docs[i].metadata["source"] for i in range(len(similar_docs))]
    similars_str = ' '.join(list_of_links)
    
    if url not in list_of_links:
        add_to_db(documents=splits)
    
    
    return f'{data}\n 3 silmilars {similars_str}'


if __name__ == '__main__':
    print(get_Chat_response('https://www.nea.com'))
