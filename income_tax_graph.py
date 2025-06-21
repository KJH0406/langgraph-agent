# %%
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name="income_tax_collection",
    persist_directory="income_tax_collection"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# %%
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str


graph_builder = StateGraph(AgentState)

# %%
def retrieve(state: AgentState):
    query = state["query"]
    docs = retriever.invoke(query)
    return {"context": docs}

# %%
from langchain_openai import ChatOpenAI

generate_llm = ChatOpenAI(model="gpt-4o", max_tokens=100)

# %%
from langchain import hub

generate_prompt = hub.pull("rlm/rag-prompt")

def generate(state: AgentState):
    context = state["context"]
    query = state["query"]
    rag_chain = generate_prompt | generate_llm 
    response = rag_chain.invoke({"context": context, "question": query})
    print(f"generate response: {response}")
    return {"answer": response.content}

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# %%
from langchain import hub
from typing import Literal

relevence_doc_prompt = hub.pull("langchain-ai/rag-document-relevance")

def check_relevence_doc(state: AgentState) -> Literal["relevant", "irrelevant"]:
    query = state["query"]
    context = state["context"]
    relevence_chain  = relevence_doc_prompt | llm 
    response = relevence_chain.invoke({"documents": context, "question": query})
    if response['Score'] == 1:
        return 'relevant'
    
    return 'irrelevant'


# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ['사람과 관련된 표현 -> 거주자']

rewrite_prompt = PromptTemplate.from_template(
"""
사용자의 질문을 보고, 우리의 사전을 참고하여 사용자의 질문을 변경해주세요.
사전 : {dictionary}
사용자의 질문 : {{query}}
"""
)

def rewrite(state: AgentState):
    query = state["query"]
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    
    response = rewrite_chain.invoke({"query": query})
    return {"query": response}

# %%
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

hallucination_llm = ChatOpenAI(model="gpt-4o", temperature=0)

hallucination_prompt = PromptTemplate.from_template(
"""
you are a teacher tasked with checking if a student's answer is based on the given documents or not.
Given the student's answer and the documents, please determine if the student's answer is based on the documents or not.
if the student's answer is not based on the documents, please return 'hallucinated'
if the student's answer is based on the documents, please return 'not hallucinated'

documents : {documents}
student_answer : {student_answer}
"""
)

def check_hallucination(state: AgentState) -> Literal["hallucinated", "not hallucinated"]:
    answer = state["answer"]
    context = state["context"]
    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    response = hallucination_chain.invoke({"student_answer": answer, "documents": context})
    return response


# %%
from langchain import hub

helpfullness_prompt = hub.pull("langchain-ai/rag-answer-helpfulness")

def check_helpfulness_grader(state: AgentState):
    query = state["query"]
    answer = state["answer"]
    helpfullness_chain = helpfullness_prompt | llm
    response = helpfullness_chain.invoke({"question": query, "student_answer": answer})
    if response['Score'] == 1:
        return 'helpful'
    else:
        return 'unhelpful'

def check_helpfulness(state: AgentState):
    return state


# %%
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("rewrite", rewrite)
graph_builder.add_node("check_helpfulness", check_helpfulness)

# %%
from langgraph.graph import START, END

# 노드 연결
graph_builder.add_edge(START, "retrieve")
graph_builder.add_conditional_edges("retrieve", check_relevence_doc, {
  "relevant": "generate",
  "irrelevant": END
})
graph_builder.add_conditional_edges("generate", check_hallucination, {
  "hallucinated": "generate",
  "not hallucinated": "check_helpfulness"
})
graph_builder.add_conditional_edges("check_helpfulness", check_helpfulness_grader, {
  "helpful": END,
  "unhelpful": "rewrite"
})
graph_builder.add_edge("rewrite", "retrieve")

# %%
# 그래프 컴파일
graph = graph_builder.compile()


