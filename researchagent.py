import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

# === Load Environment Variables ===
try:
    load_dotenv()
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
except Exception as e:
    print("[ERROR] Failed to load environment variables:", e)
    exit(1)

# === Initialize Tools ===
try:
    search_tool = SerperDevTool()
except Exception as e:
    print("[ERROR] Failed to initialize SerperDevTool:", e)
    exit(1)

def create_research_agent():
    print("[INFO] Creating research agent...")
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        return Agent(
            role="Research Specialist",
            goal="Conduct thorough research on given objects",
            backstory="You are an experienced researcher with expertise in finding and synthesizing information from various sources",
            verbose=True,
            allow_delegation=False,
            tools=[search_tool],
            llm=llm
        )
    except Exception as e:
        print("[ERROR] Failed to create research agent:", e)
        raise

def create_research_task(agent, topic):
    print(f"[INFO] Creating research task for topic: {topic}")
    try:
        return Task(
            description=f"Research the following topic and provide a comprehensive summary: {topic}",
            agent=agent,
            expected_output="A detailed summary of the research findings, including key points and insights related to the topic"
        )
    except Exception as e:
        print("[ERROR] Failed to create research task:", e)
        raise

def run_research(topic):
    try:
        agent = create_research_agent()
        task = create_research_task(agent, topic)
        crew = Crew(agents=[agent], tasks=[task])
        print("[INFO] Running research crew...")
        result = crew.kickoff()
        return result
    except Exception as e:
        print("[ERROR] Research process failed:", e)
        return None

if __name__ == "__main__":
    print("=== Welcome to the Research Agent ===")
    try:
        topic = input("Enter the research topic: ")
        print("[INFO] Starting research...")
        result = run_research(topic)
        print("\n=== Research Result ===")
        print(result if result else "No result returned.")
    except Exception as e:
        print("[ERROR] Unexpected error:", e)