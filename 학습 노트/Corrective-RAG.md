> 출처: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/#llms

## 1. Corrective RAG란?

![image.png](/images/corrective-rag-flow-chart.png)

- Corrective RAG는 RAG 전략 중 하나로, 검색된 문서에 대해 관련성 평가를 수행하고, 그 결과에 따라 필요하면 웹 검색(Web Search) 을 통해 문서를 보완하는 프로세스를 포함한다.
- 전체 흐름 요약
  1. 관련성 평가
     - 검색된 문서 중 하나라도 임계값을 초과하는 관련성이 있으면, 생성(Generation) 단계로 진행
  2. 지식 정제(Knowledge Refinement)
     - 문서를 여러 개의 지식 단편(Knowledge Strips) 으로 나누고,
     - 각 단편을 평가하여 관련성이 낮은 부분은 필터링
  3. 웹 검색 보완
     - 모든 문서가 관련성이 부족하거나, LLM이 확신을 갖지 못하면,
     - 웹 검색(Web Search) 을 통해 외부 정보를 가져옴

---

## 2. Self-RAG와의 차이점

- Self-RAG에서는 질문으로 적절한 문서를 검색할 수 없다면, Rewrite를 통해 질문을 개선한 후 다시 Retrieval을 수행한다.
- 하지만 질문 자체가 벡터 DB와 전혀 관련이 없다면, Rewrite → Retrieve의 루프가 무한 반복될 수 있는 한계가 있었다.
  - 예시: 소득세 문서를 기반으로 한 벡터 DB에서 "판교 맛집"을 계속 검색하려는 경우,검색 자체가 불가능하며, Rewrite만으로는 문제를 해결할 수 없음
- 여기서 Corrective RAG는 관련 없는 문서가 반환될 경우, 질문을 다시 쓰는 것이 아니라 Web Search를 수행하여 외부 정보 소스를 활용할 수 있다.
- 구조 자체는 Self-RAG와 유사하며, Web Search 노드 하나만 추가하여 구현 가능하기 때문에 보다 쉽게 구성할 수 있었다.
- 활용 아이디어
  - 향후에는 검색된 문서의 관련성이 낮은 경우, 곧바로 웹 검색을 수행하기보다는 먼저 질문을 Rewrite하여 검색 쿼리를 개선한 뒤 다시 한 번 검색을 시도하고, 그 결과 역시 관련성이 충분하지 않을 경우 웹 검색(Web Search)을 하는 식으로 Self-RAG와 Corrective개념을 병합하여 활용하면 좋을 것 같다.

---

## 3. 튜토리얼 코드에서 발견한 문제

튜토리얼에서의 `decide_to_generate` 함수 주석에는 모든 문서가 관련성이 없는 경우 쿼리를 재작성한다고 되어있었다. 하지만 실제 로직은 **문서들 중 하나라도 관련성이 없을 경우** `web_search = "Yes"`가 되는 식으로 동작하고 있었다.

```python
if web_search == "Yes":
    # All documents are not relevant to question
    print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")

```

### ✅ 기존 구현

```python
filtered_docs = []
web_search = "No"

for d in documents:
    score = retrieval_grader.invoke({"question": question, "document": d.page_content})
    grade = score.binary_score
    if grade == "yes":
        filtered_docs.append(d)
    else:
        web_search = "Yes"
        continue

```

이처럼 관련 문서가 있어도, 검색된 문서 중 하나라도 관련성이 없으면 `web_search`가 `"Yes"`로 설정된다.

### ✅ 내가 수정한 코드

```python
filtered_docs = []
web_search = "No"

for d in documents:
    score = retrieval_grader.invoke({"question": question, "document": d.page_content})
    grade = score.binary_score
    if grade == "yes":
        print("---GRADE: DOCUMENT RELEVANT---")
        filtered_docs.append(d)
    else:
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        continue

# 모든 문서가 관련성이 없을 때만 web search = "Yes"로 설정
if len(filtered_docs) == 0:
    web_search = "Yes"

return {"documents": filtered_docs, "question": question, "web_search": web_search}
```

---

## 4. 마치며

생각보다 튜토리얼 코드에 오류가 많은 것 같다. 그래도 덕분에(?) 더 꼼꼼하게 코드를 뜯어보게 되었던 것 같다.

이번 학습을 통해 유의미 했던 것, LLM 활용 방식의 변화에 대해서 흐름을 명확하게 이해하게 되었다는 점이다.

결국 LLM은 기존에 학습된 지식만으로는 응답의 정확성을 완전히 보장할 수 없고, 실제로 학습되지 않은 지식은 응답할 수 없는 한계를 가지고 있었다.

이러한 한계를 보완하기 위해 나온 개념이 바로 검색 증강 생성(RAG) 이고, 여기서도 더 나아가, 검색된 문서의 관련성에 지나치게 의존하거나 검색이 잘못되었을 경우 어떻게 대응할 것인가에 대한 문제에 대해서

검색된 문서가 부적절할 경우, 단순히 재검색을 반복하는 것이 아니라, 웹 검색(web search) 을 통해 외부 정보를 활용하는 Corrective RAG 방식까지 익힐 수 있었다.

특히 직접 코드를 구현하면서 이런 발전 흐름을 하나씩 체감할 수 있었던 점이 좋았다.

LangGraph 튜토리얼은 관련 논문을 일부 반영해 만든 예제라서 아직 완전한 개념 전체를 코드로 구현한 수준은 아니지만, 기초 개념을 확실히 익힌 뒤에는 실제 논문 수준의 로직도 직접 구현해 보면서 더 깊이 있게 이해해봐야겠다는 생각이 들었다.
