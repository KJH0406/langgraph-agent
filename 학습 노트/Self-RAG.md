> 출처 : https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag

---

## 1. Self-RAG란?

- Self-RAG는 검색 기반 생성(RAG)에 자기 반성(Self-reflection) / 자기 평가(Self-grading)를 도입한 전략이다.
- 단순히 검색 → 생성 흐름이 아니라, **(1) 검색이 적절했는가**, **(2) 생성이 근거에 기반했는가**, **(3) 생성 결과가 질문에 유용했는가** 등을 **스스로 판단**하게 하여 **정확도와 신뢰도**를 높이는 전략이다.
- 각 단계의 의미 분석
  | 단계 | 의도 | 설명 |
  | --------------------------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------- |
  | **1. Should I retrieve from retriever, R** | **검색 필요성 판단** | 질문(x)에 대해 외부 정보를 검색해야 할지, 기존 학습된 정보로 충분할지 결정 |
  | **2. Input: x (question) OR x, y** | **결정의 입력값** | 오리지널 질문(x) 또는 질문과 기존 생성 결과(y)를 바탕으로 판단할 수 있음. 예: "이미 충분히 답변했는가?" |
  | **3. Decides when to retrieve D chunks with R** | **검색 시점 결정** | 정보 부족 시 검색기(R)를 통해 D개의 문서 조각을 검색 |
  | **4. Output: yes, no, continue** | **의사 결정 결과** | 검색할지 말지 또는 계속 이전 정보로 진행할지를 반환 |
  | **5. Are the retrieved passages D relevant to the question x** | **검색 결과의 관련성 평가** | 검색된 각 문서 청크(d)가 질문(x)에 유의미한 정보를 제공하는지 평가합니다 |
  | **6. Input: (x, d) for d in D** | **각 문서 조각 평가** | 질문과 각 청크를 페어로 입력하여 개별 평가를 수행 |
  | **7. d provides useful information to solve x** | **평가 기준** | d가 x를 해결하는 데 실질적인 도움이 되는지를 기준으로 판단 |
  | **8. Output: relevant, irrelevant** | **출력 결과** | 각 청크에 대해 관련 있음 / 없음으로 결과를 분류 |
  | **9. Are the LLM generation from each chunk in D is relevant to the chunk** | **생성 응답의 근거 정당성 평가** | 모델이 해당 문서를 바탕으로 올바른 응답을 생성했는지 평가(hallucination)이 없는지 |
  | **10. Input: x, d, y for d in D** | **평가 입력** | 질문(x), 문서 청크(d), 생성된 응답(y)을 함께 넣어 평가 |
  | **11. All of the verification-worthy statements in y are supported by d** | **사실 검증 기준** | 응답 y에 포함된 중요한 주장들이 d에서 근거를 찾을 수 있어야 함 |
  | **12. Output: {fully supported, partially supported, no support}** | **지원 정도 평가** | 완전히, 부분적으로, 또는 전혀 지원되지 않는지로 분류(\* 이는 hallucination 정도를 반영) |
  | **13. The LLM generation from each chunk in D is a useful response to x** | **응답의 질문 해결력 평가** | 해당 응답이 실제 질문에 잘 대응했는지를 주관적으로 평가 |
  | **14. Input: x, y for d in D** | **입력** | 질문과 생성된 응답을 기준으로 판단 |
  | **15. y is a useful response to x** | **효용성 판단 기준** | 단순히 관련성뿐 아니라 실제로 유용하고 실질적인 답변인지 평가 |
  | **16. Output: {5, 4, 3, 2, 1}** | **정성 평가** | 높은 점수일수록 더 유용하다는 의미의 만족도 평가 |

---

## 2. Self-RAG 플로우 다이어그램

![](https://velog.velcdn.com/images/usergonggonggong/post/876a3698-63d6-4c07-af22-71996cf4e17c/image.png)

> ❗여기서 주의해야할 점
> → 다이어그램 내에서는 “Docs relevant?” 에서 “No”이면 마치 흐름이 종료되는 것으로 표현되어있는데, 실제 구현된 코드를 보면 **Rewrite** 노드를 실행하도록 간선(Edge)가 이어져 있다. 결국 검색된 문서가 질문과 관련이 없다고 판단되면, **질문을 재작성(Rewrite)하고 검색(Retrieve)을 다시 시도하는 흐름**이 이어지게 된다.
> (내가 뭘 놓쳤나? 싶어 코드 한 줄 한 줄 다시 읽고 검색하느라 정리에 오래걸렸다…)

1. 사용자가 질문을 입력 (**Question = query**)
2. 입력한 질문을 기반으로 벡터 데이터베이스에서 관련 문서를 검색 (**Retrieve**)
3. 검색된 문서가 질문과 충분히 관련 있는지 판단 (**Docs relevant?**)
   1. 관련성이 높음 → 해당 문서를 바탕으로 답변 생성 (**Generate**)
   2. 관련성이 낮음 → 질문을 다시 재작성하여 반복 (**Rewrite → Retrieve**)
4. 생성된 답변이 검색된 문서를 기반으로 한 것인지 판단 (**Hallucinations?**)
   1. 문서를 기반으로 생성되었음 (환각 X) → 다음 단계로 진행
   2. 문서를 기반으로 하지 않았음 (환각 O) → 질문을 다시 재작성하여 반복 (**Rewrite → Retrieve**)
5. 생성된 답변이 질문에 명확히 응답하는지 (= 사용자에게 유용한지) 판단 (**Answers question?**)
   1. 질문에 적절하게 응답함 → 최종 답변 반환 및 종료 (**Answer & END**)
   2. 질문에 적절하지 않음 → 질문을 재작성하고 반복 (**Rewrite → Retrieve**)

---

## 3. 코드 구현 요약

1. 문서 임베딩 및 검색기 구성 (Retriever)
2. LLM 구성(Grade_documentsGenerate etc... )
3. 그래프 상태 정의(State)
4. 그래프 노드 생성(Nodes)
5. 그래프 간선 연결(Edges)
6. 그래프 컴파일 및 시각화(Compile & Display)
7. 그래프 실행(Run)

---

### 3-1. 문서 임베딩 및 검색기 구성 (Retriever)

```python
# Retriever(검색기)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 참고 사이트(데이터 소스)
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 문서 불러오기
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# 벡터 데이터베이스 생성
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()
```

- `WebBaseLoader` 를 활용하여 인터넷 웹사이트 데이터 추출
- [RecursiveCharacterTextSplitter](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html) 를 활용하여 텍스트를 문서 단위로 분할
- 분할된 문서를 기반으로 벡터 데이터베이스 생성 및 검색기 생성

---

### 3-2. LLMs 구성

```python
# Retrieval Grader (문서 검색 정확도 평가)

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field

# 문서 관련성 평가
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 프롬프트
system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

question = "agent memory"  # 사용자 질문 입력
docs = retriever.invoke(question)  # 질문을 기반으로 검색기를 통해 검색된 문서 가져오기
doc_txt = docs[0].page_content  # 검색된 문서 중 첫 번째 문서의 내용 가져오기
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))  # 검색된 문서가 사용자 질문과 관련이 있는지 평가하기
```

- **검색된 문서가 사용자 질문과 관련이 있는지 평가**하는 역할
- 사용자의 질문(question)을 입력받고, 벡터 데이터베이스 검색(retriever)를 통해서 검색된 문서를 받아옴
- 검색된 문서와 사용자의 질문을 입력받고, 검색된 문서가 사용자 질문과 관련이 있는지 평가한 후 **'yes' or 'no'로 응답**

---

```python
# Generate (답변 생성)
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt | llm | StrOutputParser()

context = format_docs(docs)
generation = rag_chain.invoke({"context": context, "question": question})
print(generation)
```

- **검색된 문서가 사용자 질문과 관련이 있다면 답변 생성**
- 사용자의 질문(question)과 Rretreive된 문서를 기반으로 질문에 대한 답변 생성

---

```python
# Hallucination Grader (생성된 답변에 허위 정보가 포함되어 있는지 평가)

# 데이터 모델
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
hallucination_grader.invoke({"documents": docs, "generation": generation})
```

- 생성된 답변에 대하여 **할루시네이션 평가하는 역할 수행**
- 생성된 답변과 문서를 비교하여 **생성된 답변이 문서를 기반**으로 응답했는지 확인

---

```python
# Answer Grader (생성된 답변이 사용자 질문을 해결하는지 평가)

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
answer_grader.invoke({"question": question, "generation": generation})
```

- 만약 할루시네이션 단계까지 통과했다면, **최종적으로 생성된 답변이 사용자 질문의 유의미한 답변을 했는지 평가**
- 질문과 생성된 답변을 비교하여, **사용자 질문에 적절하게 답변하였다면 그대로 출력 후 실행 종료. 만약 적절하게 답변하지 않았다면 질문을 재작성(Rewrite)하고 다시 문서를 검색(Retrieve)**

---

```python
# Question Re-writer (사용자 질문을 더 좋은 질문으로 변환)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})
```

- 사용자의 **질문을 다시 재작성하는 역할**
- 검색 결과로 반환된 문서가 관련성이 낮아 질문을 다시 Rewrite하거나, 최종적으로 생성된 답변이 사용자 질문에 적절하게 응답하지 않을 경우에 실행

---

### 3-3. 그래프 상태 정의(State)

```python
from typing import List

from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
```

- 사용자의 질문과 생성 결과, 검색된 문서 관리

---

### 3-4. 그래프 노드 생성(Nodes)

```python
# 노드 생성

# 문서 검색 노드
# 사용자의 질문을 기반으로 벡터 데이터 베이스에서 관련 문서 추출
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # 문서 검색
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

# 답변 생성 노드
# 사용자 질문, 관련성 있는 문서(grade_documemts 통과한 문서)를 기반으로 답변 생성
def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG 답변 생성
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

# 문서 검색 관련성 평가 노드
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # 각 문서에 대한 점수 계산 및 포함할 문서 필터링
    filtered_docs = []
    # Retrieve된 문서 리스트 확인
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content} # 페이지 내용만 판단
        )
        grade = score.binary_score # yes or no로 출력
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            # 관련 있는 문서만 docs 배열에 추가
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question} # 관련성 있는 문서만 포함

# 질문 재작성 노드
def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # 질문 재작성
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}
```

- 3-2 단계에서 정의한 LLM을 기반으로 노드 생성
- 문서 검색 노드, 답변 생성 노드, 문서 검색 관련성 평가 노드, 질문 재작성 노드

---

### 3-5. 그래프 간선 연결(Edges)

```python
# 간선 생성

# 답변 생성 여부 결정
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    # 모든 문서가 필터링되었으면(= 관련 문서가 없으면) 질문 재작성
    if not filtered_documents:
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # 관련 문서가 있으면 답변 생성
        print("---DECISION: GENERATE---")
        return "generate"

# 답변 생성 정확도 평가
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # 허위 정보 포함 여부 확인
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---") # 답변이 문서에 근거가 있음
        # 질문 해결 여부 확인
        print("---GRADE GENERATION vs QUESTION---") # 답변이 질문을 해결하는지 평가
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---") # 답변이 질문을 해결함
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---") # 답변이 질문을 해결하지 못함
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---") # 답변이 문서에 근거가 없음
        return "not supported"
```

- decide_to_generate = 답변 생성 여부 결정
  - 관련 문서가 1개라도 있으면 답변 생성 노드(generate)로 연결
  - 관련 문서가 1개도 없다면 질문 재작성 노드로 연결(transform_query)
- grade_generation_v_documents_and_question = 답변 생성 정확도 평가
  - 허위 정보 포함 여부 확인(hallucination_grader)하여 생성한 답변이 문서에 기반하지 않았으면 답변 재생성을 위해 (generate)로 연결
  - 허위 정보가 없다면, 생성된 답변이 질문의 유의미하게 답변했는지 확인(answer_grader)하고 유의미하면 종료, 유의미하지 않다면 질문 재작성 노드로 연결(transform_query)하여 다시 문서 검색부터 수행

---

### 3-6. 그래프 컴파일 및 시각화(Compile & Display)

```python
# 그래프 생성 및 간선 연결

from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# 노드 생성
workflow.add_node("retrieve", retrieve)  # 문서 검색
workflow.add_node("grade_documents", grade_documents)  # 문서 검색 관련도 평가
workflow.add_node("generate", generate)  # 답변 생성
workflow.add_node("transform_query", transform_query)  # 질문 재작성

# 간선 연결
workflow.add_edge(START, "retrieve") # 문서 검색
workflow.add_edge("retrieve", "grade_documents") # 문서 관련성 평가
# 문서 관련성 여부에 따라서 조건부 분기
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,# 답변 생성 여부 결정
    {
        "transform_query": "transform_query", # 질문 재작성
        "generate": "generate", # 답변 생성
    },
)
workflow.add_edge("transform_query", "retrieve") # 질문 재작성 후 문서 검색

# 답변이 할루시네이션이 발생했는지 사용자 질문의 유의미한 답변을 했는지에 따라서 조건부 분기
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate", # 답변이 문서에 근거가 없음
        "useful": END, # 답변이 문서에 근거가 있음
        "not useful": "transform_query", # 답변이 질문을 해결하지 못함
    },
)

# 컴파일
app = workflow.compile()
```

![image.png](/images/self-rag-flow-chart.png)

**_Self-RAG 워크 플로우_**

1. 시작 (START)
2. 입력한 질문을 기반으로 벡터 데이터베이스에서 관련 문서를 검색 (retrieve)
3. 검색된 문서가 질문과 충분히 관련 있는지 판단(grade_documents / decide_to_generate)
   1. 관련 문서 있음 → 해당 문서를 바탕으로 답변 생성
   2. 관련 문서 없음 → 질문을 다시 재작성하여 반복
4. 생성된 답변이 검색된 문서를 기반으로 한 것인지 판단 (grade_generation_v_documents_and_question)
   1. 문서를 기반으로 생성되었음 (환각 X) → 5단계 진행
   2. 문서를 기반으로 하지 않았음 (환각 O) → 질문을 다시 재작성하여 반복 (**Rewrite → Retrieve**)
5. 생성된 답변이 질문에 명확히 응답하는지 (= 사용자에게 유용한지) 판단 (**Answers question?**)
   1. 질문에 적절하게 응답함 → 최종 답변 반환 및 종료 (**Answer & END**)
   2. 질문에 적절하지 않음 → 질문을 재작성하고 반복 (**Rewrite → Retrieve**)

### 3-7. 그래프 실행(Run)

---

```python
from pprint import pprint

inputs = {"question": "Explain how chain of thought prompting works?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])
```
