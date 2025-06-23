from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import Field
from crewai_tools import PDFSearchTool
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
import requests
import os

# === Load Environment Variables ===
try:
    load_dotenv()
    os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
except Exception as e:
    print("[ERROR] Failed to load environment variables:", e)
    exit(1)

# === Download PDF ===
pdf_url = 'https://arxiv.org/pdf/2405.01577'
pdf_filename = 'hatespeech.pdf'
try:
    print(f"[INFO] Downloading PDF from {pdf_url} ...")
    response = requests.get(pdf_url)
    response.raise_for_status()
    with open(pdf_filename, 'wb') as file:
        file.write(response.content)
    print(f"[INFO] PDF saved as {pdf_filename}")
except Exception as e:
    print("[ERROR] Failed to download or save PDF:", e)
    exit(1)

# === Initialize LLM ===
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo")
except Exception as e:
    print("[ERROR] Failed to initialize LLM:", e)
    exit(1)

# === Define Agents ===
try:
    Router_Agent = Agent(
        role='Router',
        goal='Route user question to a vector store, web search, or LLM generation',
        backstory=(
            "You are an expert at routing a user question to the correct tool. "
            "If the question is about the provided PDF or its topic, use the vectorstore tool. "
            "If the question is about current news, events, or things not in the PDF, use the web search tool. "
            "For general knowledge or creative questions, use the LLM generation tool."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    Retriever_Agent = Agent(
        role='Retriever',
        goal="Use the information retrieved from the vectorstore to answer the question",
        backstory=(
            "You are an assistant for question-answering tasks."
            "Use the information present in the retrieved context to answer the question."
            "You have to provide a clear concise answer within 200 words."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
except Exception as e:
    print("[ERROR] Failed to create agents:", e)
    exit(1)

# === Define Tools ===
try:
    search = GoogleSerperAPIWrapper()
    class SearchTool(BaseTool):
        name: str = "Search"
        description: str = "Useful for search-based queries. Use this to find current information about markets, companies, and trends."
        search: GoogleSerperAPIWrapper = Field(default_factory=GoogleSerperAPIWrapper)
        def _run(self, query: str) -> str:
            try:
                return self.search.run(query)
            except Exception as e:
                return f"Error performing search: {str(e)}"
    class GenerationTool(BaseTool):
        name: str = "Generation_tool"
        description: str = "Useful for generic-based queries. Use this to find information based on your own knowledge."
        def _run(self, query: str) -> str:
            try:
                llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
                return llm.invoke(query)
            except Exception as e:
                return f"Error generating response: {str(e)}"
    generation_tool = GenerationTool()
    web_search_tool = SearchTool()
    pdf_search_tool = PDFSearchTool(pdf=pdf_filename)
except Exception as e:
    print("[ERROR] Failed to initialize tools:", e)
    exit(1)

# === Define Tasks ===
try:
    router_task = Task(
        description=("Analyse the keywords in the question {question} "
            "Based on the keywords decide whether it is eligible for a vectorstore search or a web search or generation. "
            "Return a single word 'vectorstore' if it is eligible for vectorstore search. "
            "Return a single word 'websearch' if it is eligible for web search. "
            "Return a single word 'generate' if it is eligible for generation. "
            "Do not provide any other preamble or explanation."
        ),
        expected_output=("Give a choice 'websearch' or 'vectorstore' or 'generate' based on the question. "
            "Do not provide any other preamble or explanation."),
        agent=Router_Agent,
    )
    retriever_task = Task(
        description=("Based on the response from the router task extract information for the question {question} with the help of the respective tool. "
            "Use the web_search_tool to retrieve information from the web in case the router task output is 'websearch'. "
            "Use the rag_tool to retrieve information from the vectorstore in case the router task output is 'vectorstore'. "
            "Otherwise generate the output based on your own knowledge in case the router task output is 'generate'."
        ),
        expected_output=("You should analyse the output of the 'router_task'. "
            "If the response is 'websearch' then use the web_search_tool to retrieve information from the web. "
            "If the response is 'vectorstore' then use the rag_tool to retrieve information from the vectorstore. "
            "If the response is 'generate' then use generation_tool. "
            "Otherwise say 'I don't know' if you don't know the answer. "
            "Return a clear and concise text as response."),
        agent=Retriever_Agent,
        context=[router_task],
        tools=[pdf_search_tool, web_search_tool, generation_tool],
    )
except Exception as e:
    print("[ERROR] Failed to create tasks:", e)
    exit(1)

# === Create Crew and Run ===
def run_rag_crew():
    try:
        rag_crew = Crew(
            agents=[Router_Agent, Retriever_Agent],
            tasks=[router_task, retriever_task],
            verbose=True,
        )
        print("[INFO] RAG Crew created successfully.")
        question = input("Enter your question: ")
        print(f"[INFO] Running RAG crew for question: {question}")
        result = rag_crew.kickoff(inputs={"question": question})
        print("\n=== RAG Crew Result ===")
        print(result)
    except Exception as e:
        print("[ERROR] RAG Crew execution failed:", e)

if __name__ == "__main__":
    print("=== RAG Pipeline ===")
    run_rag_crew()


