import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

os.environ["SERPER_API_KEY"] = ""  # serper.dev API key
os.environ["OPENAI_API_KEY"] = ""

# You can pass an optional llm attribute specifying what model you wanna use.
# It can be a local model through Ollama / LM Studio or a remote
# model like OpenAI, Mistral, Antrophic or others (https://docs.crewai.com/how-to/LLM-Connections/)
#
# import os
# os.environ['OPENAI_MODEL_NAME'] = 'gpt-3.5-turbo'
#
# OR
#
# from langchain_openai import ChatOpenAI

search_tool = SerperDevTool()






# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science in healtcare',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=False,
  # You can pass an optional llm attribute specifying what model you wanna use.
  # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7),
  tools=[search_tool]
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist for linkedin , known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in generative AI and the use of agents.
  Identify key trends, breakthrough technologies, and potential industry impacts.""",
  expected_output="Full analysis report in bullet points",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.Write it in the most human like way possible.
  Make a top of the world linked in post in order to have the most number of reactions posible""",
  expected_output="A linked in post",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2, # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

#print("######################")
#print(result)

#display(Markdown(result))
