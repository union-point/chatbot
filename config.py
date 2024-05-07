from dotenv import load_dotenv
import os
load_dotenv()

configs = {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
           "db_path": os.path.join(os.path.dirname(os.path.abspath(__file__)),'db'),
           "db_name": "vectorstore",
           }
