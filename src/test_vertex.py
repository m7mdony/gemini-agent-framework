from vertex_agent import Agent

agent = Agent(key_path="/home/arete/capstone/gemini-agent-framework/1.json", model_name="gemini-2.0-flash", region="us-central1")

agent.set_project(key_path="/home/arete/capstone/gemini-agent-framework/2.json")

response = agent.prompt(
    user_prompt="hello",
    system_prompt=" you are a helpful assistant"
)

print(response)