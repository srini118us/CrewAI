import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# === Load Environment Variables ===
try:
    load_dotenv()
    os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["OPEN_AI_MODEL"] = "gpt-4-32k"
except Exception as e:
    print("[ERROR] Failed to load environment variables:", e)
    exit(1)

# === Initialize Tools ===
try:
    search_tool = SerperDevTool()
except Exception as e:
    print("[ERROR] Failed to initialize SerperDevTool:", e)
    exit(1)

# === Define Agents ===
try:
    researcher = Agent(
        role='Senior Researcher',
        goal='Uncover groundbreaking technologies in {topic}',
        verbose=True,
        memory=True,
        backstory=(
            "Driven by curiosity, you're at the forefront of innovation,"
            "eager to explore and share knowledge that could change the world."
        ),
        tools=[search_tool],
        allow_delegation=False
    )

    writer = Agent(
        role='Writer',
        goal='Narrate compelling tech stories about {topic}',
        verbose=True,
        memory=True,
        backstory=(
            "With a flair for simplifying complex topics,"
            "you craft engaging narratives that captivate and educate,"
            "bringing new discoveries to light in an accessible manner."
        ),
        tools=[search_tool],
        allow_delegation=False
    )
except Exception as e:
    print("[ERROR] Failed to create agents:", e)
    exit(1)

# === Define Tasks ===
try:
    research_task = Task(
        description=(
             "Identify the next big trend in {topic}. "
            "Focus on identifying pros and cons and the overall narrative. "
            "Your final report should clearly articulate the key points, "
            "its market opportunities, and potential risks."
        ),
        expected_output="A comprehensive 3 paragraphs long report on the latest AI trends.",
        tools=[search_tool],
        agent=researcher
    )

    write_task = Task(
        description=(
            "Compose an insightful article on {topic}. "
            "Focus on the latest trends and how it's impacting the industry. "
            "This article should be easy to understand, engaging, and positive."
        ),
        expected_output="A 4 paragraph article on {topic} advancements formatted as markdown.",
        tools=[search_tool],
        agent=writer,
        async_execution=False,
        output_file="new-blog-post.md"
    )
except Exception as e:
    print("[ERROR] Failed to create tasks:", e)
    exit(1)

# === Create Crew and Run ===
def run_crew():
    try:
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            verbose=True,
            process=Process.sequential
        )
        print("[INFO] Crew created successfully.")
        topic = input("Enter topic: ")
        print("[INFO] Running crew for topic:", topic)
        result = crew.kickoff({"topic": topic})
        print("\n=== Crew Result ===")
        print(result)
    except Exception as e:
        print("[ERROR] Crew execution failed:", e)

if __name__ == "__main__":
    print("=== CrewAI Tech Trend Analyzer ===")
    run_crew()

