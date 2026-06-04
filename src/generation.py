from src.utility import get_llm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import Literal

REWRITE_PROMPT = (
    "Analyze the user's question and conversation history.\n"
    "Respond in one of two formats only:\n\n"
    "If greeting, social, or no document search needed:\n"
    "  action: skip, query: ''\n\n"
    "If document question that needs search:\n"
    "  action: retrieve, query: <rewritten standalone question>"
)

SYSTEM_PROMPT = (
    "You are DocLens, an intelligent document analysis assistant.\n"
    "Answer using ONLY the provided context documents.\n"
    "If context is insufficient, say so clearly.\n"
    "Be concise and cite specifics from the context when possible."
)

# 1. Load LLM
def load_llm() -> ChatGoogleGenerativeAI:
    print("Loading LLM model")
    try:
        llm = get_llm()
        print(f"Loaded LLM model : {llm}")
        return llm
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        return None


# 2. Rewrite User Query
class QueryPlan(BaseModel):
    action: Literal["skip", "retrieve"]
    query: str = ""

def rewrite_query(llm: ChatGoogleGenerativeAI, chat_history: list, query: str) -> QueryPlan:
    # 2.1. Build prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])

    # 2.2. Bind structured output to LLM
    print(f"Binding structured output to LLM with QueryPlan schema")
    structured_llm = llm.with_structured_output(QueryPlan)
    
    # 2.3. Build and invoke chain
    print(f"Building chain with prompt and structured LLM")
    chain = prompt | structured_llm
    
    #2.4. Invoke chain and get query plan
    print(f"Invoking chain to rewrite query with chat_history: {chat_history} and question: {query}")
    try:
         plan = chain.invoke({"chat_history": chat_history, "question": query})
         print(f"Received query plan from LLM: action={plan.action}, query={plan.query}")
         return plan
    except Exception as e:
        print(f"Error rewriting query: {e}")

# 3. Generate Answer
def generate_answer(llm: ChatGoogleGenerativeAI, chat_history: list, query: str, docs: list) -> str:
    # 3.1. Build context string from retrieved docs
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3.2. Build prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    # 3.3. Bind LLM to prompt
    print(f"Binding LLM to prompt for answer generation")   
    structured_llm = llm.with_structured_output(QueryPlan)  

    # 3.4. Build and invoke chain
    print(f"Building chain with prompt and LLM for answer generation")
    chain = prompt | structured_llm

    #3.5. Invoke chain and get answer
    print(f"Invoking chain to generate answer with context and question")
    try: 
        result = chain.invoke({"context": context, "chat_history": chat_history, "question": query})
        print(f"Received answer from LLM: {result.content}")
        return result.content
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while generating the answer."

# 4. Run Generation
def run_generation(llm: ChatGoogleGenerativeAI, chat_history: list, query: str, docs: list) -> str:
    print(f"Running generation pipeline with query: {query} and chat_history: {chat_history}")
    try: 
        plan = rewrite_query(llm, chat_history, query)
        print(f"Query plan: action={plan.action}, query={plan.query}")
        if plan.action == "skip":
            print("Skipping retrieval, generating answer directly")
            return generate_answer(llm, chat_history, query, docs)
        elif plan.action == "retrieve":
            print("Retrieval needed, generating answer with retrieved docs")
            return generate_answer(llm, chat_history, plan.query, docs)
        else:
            print(f"Unknown action in query plan: {plan.action}, defaulting to direct answer generation")
            return generate_answer(llm, chat_history, query, docs)
    except Exception as e:
        print(f"Error in generation pipeline: {e}")
        return "Sorry, I encountered an error while processing your request."

# 5. Run Generation Pipeline
def run_generation_pipeline(query: str, docs: list, chat_history: list = []) -> str:
    print(f"Starting generation pipeline for query: {query} with chat_history: {chat_history} and docs: {docs}")
    try:
        llm = load_llm()
        if llm:
          return run_generation(llm, chat_history, query, docs)
    except Exception as e:
        print(f"Error running generation pipeline: {e}")
        return "Sorry, I encountered an error while processing your request."

if __name__ == "__main__":
    answer = run_generation_pipeline("What is AGI?", docs=[])
    print(answer)