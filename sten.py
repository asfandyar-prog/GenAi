from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel





load_dotenv()

model1 = ChatGroq(
    model="openai/gpt-oss-120b"
)
model2 = ChatGroq(
    model="openai/gpt-oss-120b"
)

prompt1=PromptTemplate(
    template='Generate a simple notes from the following text \n {text} ',
    intput_variables=['text']
)
prompt2=PromptTemplate(
    template='Generate 5 short and tricky questions from  the following text \n {text}',
    intput_variables=['text'])

prompt3 =  PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes ->{notes} and {quiz}',

    intput_variables=['notes','quiz']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel(
    {'notes': prompt1| model1|parser,
    'quiz': prompt2| model2|parser}
)

merge_chain=prompt3|prompt1|parser

chain=parallel_chain|merge_chain
text="""
A Support Vector Machine (SVM) is a powerful supervised machine learning algorithm used for classification and regression that finds the optimal "hyperplane" (a decision boundary) to separate different data classes by maximizing the margin between them. SVMs excel at high-dimensional data, handle both linear and non-linear problems (using kernel tricks), and are memory-efficient as they rely on crucial "support vectors" (closest data points) to define boundaries, preventing overfitting. They are widely used in image recognition, text classification, and fraud detection.  
This video provides a quick overview of how Support Vector Machines work
"""


response=chain.invoke({'text':text})
print(response)