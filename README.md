## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
Extracting specific, nuanced information from a collection of dense academic papers is a slow and inefficient manual process. Standard search tools rely on exact keywords and fail to understand the conceptual context of a user's question. This program aims to build an AI agent that can intelligently query multiple documents to synthesize precise answers to complex questions.

### DESIGN STEPS:

#### STEP 1:
Load PDF documents and create specialized search and summary tools for each paper.

#### STEP 2:
Initialize an AI agent with an OpenAI model, giving it access to all the created tools.

#### STEP 3:
Query the agent with a specific question about one paper to get a detailed answer from its content.

### PROGRAM:
```
Name:Kasivishvanath V
reg no:212222040073
```
```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
```

```
import nest_asyncio
nest_asyncio.apply()
```

```
urls = [
    "https://openreview.net/pdf?id=agPpmEgf8C",
    "https://openreview.net/pdf?id=ekeyCgeRfC",
    "https://openreview.net/pdf?id=KS8mIvetg2",
]

papers = [
    "one.pdf",
    "two.pdf",
    "three.pdf",
]
```

```
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
```

```
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
```

```
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
```

```
len(initial_tools)
```

```
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
```
```
response = agent.query(
    "Tell me about the evaluation dataset used in two, "
    "and then tell me about the evaluation results"
)
```
```
response = agent.query("Give me a summary of both one and two")
print(str(response))
```

### OUTPUT:
<img width="1364" height="524" alt="image" src="https://github.com/user-attachments/assets/667ef194-8257-4070-92b6-b56eaabcb646" />
<img width="1351" height="918" alt="image" src="https://github.com/user-attachments/assets/0b26a2d3-069f-4bbc-a1cc-5a670a51eb29" />
<img width="1351" height="864" alt="image" src="https://github.com/user-attachments/assets/d7184732-bfdc-405a-b935-ffd03e11ba6b" />


### RESULT:
The system successfully retrieves and synthesizes relevant information from multiple documents, providing concise and relevant answers to the user's query. Performance is evaluated based on the accuracy, relevance, and coherence of the responses.
