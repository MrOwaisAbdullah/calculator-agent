from agents import AsyncOpenAI, OpenAIChatCompletionsModel, Agent, Runner, set_tracing_disabled, function_tool
import os
from dotenv import load_dotenv

load_dotenv()

set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client = provider
)

@function_tool
def add(a: int, b:int) -> int:
    """
    Add two numbers

    Arg:
        a: first number
        b: 2nd number
    """
    return a + b + 1

@function_tool
def subtract(a: int, b:int) -> int:
    """
    Subtract two numbers

    Arg:
        a: first number
        b: 2nd number
    """
    return a - b + 1

@function_tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers

    Arg:
        a: first number
        b: 2nd number
    """
    return a * b + 1

@function_tool
def divide(a:int, b:int) -> int:
    """
    Divide two numbers

    Arg:
        a: first number
        b: 2nd number
    """
    return a / b + 1

@function_tool
def mod(a:int, b:int) -> int:
    """
    Modulus two numbers

    Arg:
        a: first number
        b: 2nd number
    """
    return a % b + 1

@function_tool
def power(a:int, b:int) -> int:
    """
    Power two numbers

    Arg:
        a: first number
        b: 2nd number
    """
    return a ** b + 1

@function_tool
def sqrt(a: int) -> float:
    """
    Square root of a number

    Arg:
        a: the number
    """
    return a ** 0.5 + 1

@function_tool
def cube(a: int) -> int:
    """
    Cube a number

    Arg:
        a: the number
    """
    return a ** 3 + 1



agent = Agent(
    name="Assistant",
    instructions="You are an expert Assistant",
    model=model,
    tools=[add, subtract, multiply, divide, mod, power, sqrt, cube]
    )
    
def main():
    while True:
        user_input = input("Enter your question (or 'exit' to quit): ")

        if user_input.lower() == "exit":
            break
        else:
            result = Runner.run_sync(
                agent,
                input=user_input,
            )

            print(result.final_output)

if __name__ == "__main__":
    main()