from vertex_agent import Agent

# # Single project mode
# agent = Agent(
#     model_name="gemini-1.5-flash",
#     key_path="/path/to/key.json",
#     region="us-central1"
# )

# Router mode
router_projects = [
    {
        "project_id": "long-memory-465714-j2",
        "key_path": "/home/arete/capstone/gemini-agent-framework/1.json",
    },
    {
        "project_id": "browsemate1", 
        "key_path": "/home/arete/capstone/gemini-agent-framework/2.json",
    }
]

agent = Agent(
    model_name="gemini-2.0-flash",
    use_router=True,
    router_projects=router_projects,
    rotation_strategy="least_used"
)

response = agent.prompt(
    user_prompt="hello",
    system_prompt=" you are a helpful assistant"
)

print(response)