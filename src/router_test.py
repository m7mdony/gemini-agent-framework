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
        "key_path": "/home/arete/capstone/1.json",
        "key_dict": {""}
    },
    {
        "project_id": "browsemate1", 
        "key_path": "/home/arete/capstone/2.json",
        "key_dict": {""}
    }
]

agent = Agent(
    model_name="gemini-2.0-flash",
    use_router=True,
    router_projects=router_projects,
    use_redis=True,
    redis_url="redis://:browsemate.132456@145.79.13.160:6666/0",
    router_debug_mode=True
)

response = agent.prompt(
    user_prompt="tell me about yourself",
    system_prompt=" you are a helpful assistant"
)
print(response)