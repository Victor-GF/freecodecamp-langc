from dotenv import load_dotenv
load_dotenv()

from importlib.metadata import version
core_version = version("langchain_core")
lg_version = version("langgraph")
print(f"langchain-core version: {core_version}")
print(f"LangGraph version: {lg_version}")

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic



def main() -> None:
    llm_openai = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
    )
    response_openai = llm_openai.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatOpenAI: {response_openai}")

    llm_anthropic = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0
    )
    response_anthropic = llm_anthropic.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatAnthropic: {response_anthropic}")

    print("Setup completed!")


if __name__ == "__main__":
    main()