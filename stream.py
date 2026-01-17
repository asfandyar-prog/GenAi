from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()
model = ChatGroq()

#Schema
class Review(TypedDict):
    summery:str
    sentiment: str

structured_model=model.with_structured_output(Review)
result=structured_model.invoke("""The hardware is good, but the software is even better. I would give it 5 stars if I could.""")
print(result)