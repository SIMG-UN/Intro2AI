# Agentes de lenguaje

import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

os.environ["SERPER_API_KEY"] = "your-key"  # serper.dev API key
os.environ["OPENAI_API_KEY"] = "your-key"

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
  goal='Uncover the importance of the concept of agents in the foundations of AI and why it is important to know it',
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
  goal='Craft compelling content on tech advancements in a perfect spanish',
  backstory="""You are a renowned Content Strategist for linkedin , known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the foundational concept of agents, their role in AI.""",
  expected_output="Full analysis report in bullet points",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an incredible linked in post.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.Write it in the most human like way possible.
  Make a top of the world linked in post in order to have the most number of reactions posible in spanish, not itemize too much, include coloquial language""",
  expected_output="A linked in post in spanish",
  agent=writer
)

# Instantiate your crew with a sequential process
#crew = Crew(
#  agents=[researcher, writer],
#  tasks=[task1, task2],
#  verbose=2, # You can set it to 1 or 2 to different logging levels
#)

# Get your crew to work!
#result = crew.kickoff()

#print("######################")
#print(result)

#display(Markdown(result))



# Agentes de aprendizaje por refuerzo


import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# Crear el entorno Frozen Lake
env = gym.make('FrozenLake-v1', render_mode='rgb_array')

# Inicializar la tabla Q con ceros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Hiperparámetros
alpha = 0.8  # Tasa de aprendizaje
gamma = 0.95  # Factor de descuento
epsilon = 0.1  # Parámetro de exploración

# Función para elegir una acción
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explorar
    else:
        return np.argmax(Q[state, :])  # Explotar

# Función para obtener la representación de la política
def get_policy_map(Q):
    policy_map = []
    for i in range(env.observation_space.n):
        policy_map.append(np.argmax(Q[i, :]))
    return np.array(policy_map).reshape(4, 4)

# Función para entrenar el agente
def train_agent(episodes):
    global Q
    for _ in range(episodes):
        state = env.reset()[0]
        done = False
        while not done:
            action = choose_action(state)
            new_state, reward, done, _, _ = env.step(action)
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
            state = new_state

# Configuración de la animación
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_axis_off()

# Mapeo de acciones a flechas
action_map = {0: '←', 1: '↓', 2: '→', 3: '↑'}

# Función de inicialización para la animación
def init():
    ax.clear()
    ax.set_axis_off()
    return []

# Función de actualización para la animación
def update(frame):
    ax.clear()
    ax.set_axis_off()
    
    # Entrenar el agente por un número de episodios
    train_agent(100)
    
    # Obtener la política actual
    policy = get_policy_map(Q)
    
    # Dibujar el mapa de la política
    for i in range(4):
        for j in range(4):
            if env.desc[i][j] == b'H':
                ax.text(j, 3-i, 'H', ha='center', va='center', fontsize=24, fontweight='bold')
            elif env.desc[i][j] == b'G':
                ax.text(j, 3-i, 'G', ha='center', va='center', fontsize=24, fontweight='bold')
            else:
                ax.text(j, 3-i, action_map[policy[i, j]], ha='center', va='center', fontsize=24)
    
    ax.set_title(f'Episodio: {frame * 100}')
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.grid(True)
    
    return []

# Crear la animación
anim = FuncAnimation(fig, update, frames=100, init_func=init, blit=True, interval=200)

# Guardar la animación como GIF
anim.save('frozen_lake_rl.gif', writer='pillow', fps=5)

print("La animación se ha guardado como 'frozen_lake_rl.gif'")