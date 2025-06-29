from gemini_agent import Agent
import os
from dotenv import load_dotenv
import json
from bs4 import BeautifulSoup

load_dotenv()

class HTMLAnalyzer:
    @Agent.description("Count the number of input tags in an HTML page")
    @Agent.parameters({
        'html_content': {'type': str, 'description': 'The HTML content to analyze'}
    })
    def count_inputs(self, html_content: str) -> dict:
        soup = BeautifulSoup(html_content, 'html.parser')
        input_tags = soup.find_all('input')
        return {
            "count": len(input_tags),
            "input_types": [tag.get('type', 'unknown') for tag in input_tags]
        }

# Create HTML pages
home_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Home Page</title>
</head>
<body>
    <header>
        <nav>
            <a href="/">Home</a>
            <a href="/about">About</a>
            <a href="/contact">Contact</a>
        </nav>
    </header>
    <main>
        <h1>Welcome to Our Website</h1>
        <p>This is a simple home page with no input fields.</p>
    </main>
    <footer>
        <p>&copy; 2024 Our Website</p>
    </footer>
</body>
</html>
"""

login_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Login Page</title>
</head>
<body>
    <header>
        <nav>
            <a href="/">Home</a>
            <a href="/login">Login</a>
        </nav>
    </header>
    <main>
        <h1>Login</h1>
        <form action="/login" method="POST">
            <div>
                <label for="username">Username:</label>
                <input type="text" id="username" name="username" required>
            </div>
            <div>
                <label for="password">Password:</label>
                <input type="password" id="password" name="password" required>
            </div>
            <div>
                <input type="checkbox" id="remember" name="remember">
                <label for="remember">Remember me</label>
            </div>
            <button type="submit">Login</button>
        </form>
    </main>
    <footer>
        <p>&copy; 2024 Our Website</p>
    </footer>
</body>
</html>
"""

# Create the analyzer instance
html_analyzer = HTMLAnalyzer()

# Create the agent with our tool
agent = Agent(
    api_key=os.getenv("GEMINI_API_KEY"),
    tools=[html_analyzer.count_inputs]
)

# Store the HTML pages as variables
agent.set_variable(
    name="home_page",
    value=home_page,
    description="The HTML content of the home page",
    type_hint=str
)



# Example 1: Count inputs in home page
print("\nExample 1: Count inputs in home page")
response = agent.prompt(

    user_prompt="How many input tags are in the home page?",
    system_prompt=""""You are a tool that counts the number of input tags in an HTML page.
        make the response in this format
        {
        "count": number,
        "input_types": [input1,input2 ..... ] }
      
        """,
    json_format= True
)
print(json.dumps(response, indent=2))

# Example 2: Count inputs in login page
print("\nExample 2: Count inputs in login page")
response = agent.prompt(
    "How many input tags are in this page  "+ login_page,
     system_prompt="""
        make the response in this format
        {
        "count": number,
        "input_types": [input1,input2 ..... ] }
      
        """,
    json_format= True
)
print(json.dumps(response, indent=2)) 