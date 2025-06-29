import os
from dotenv import load_dotenv
from gemini_agent import Agent

load_dotenv()

class Calculator:
    def __init__(self):
        self.memory = 0

    @Agent.description("Multiplies a number by the stored memory value.")
    @Agent.parameters({
        'number': {'type': int, 'description': 'The number to multiply with memory'}
    })
    def multiply_with_memory(self, number: int) -> int:
        result = self.memory * number
        self.memory = result
        return result

    @Agent.description("Adds a number to the stored memory value.")
    @Agent.parameters({
        'number': {'type': int, 'description': 'The number to add to memory'}
    })
    def add_to_memory(self, number: int) -> int:
        result = self.memory + number
        self.memory = result
        return result

def test_class_methods():
    # Create a calculator instance
    calculator = Calculator()
    
    # Create an agent with the calculator methods
    agent = Agent(
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[calculator.multiply_with_memory, calculator.add_to_memory]
    )

    # Test using class methods
    response = agent.prompt(
        "Multiply 5 with memory (starting at 0), then add 10 to the result",
    )
    print(response)

if __name__ == "__main__":
    test_class_methods() 