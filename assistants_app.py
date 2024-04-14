from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
import yfinance
import json
import openai as client
import streamlit as st
import time
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, DuckDuckGoSearchResults
from typing import Type


def get_ticker(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    company_name = inputs["company_name"]
    return ddg.run(f"Ticker symbol of {company_name}")


def get_income_statement(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.income_stmt.to_json())


def get_balance_sheet(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.balance_sheet.to_json())


def get_daily_stock_performance(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.history(period="3mo").to_json())


functions_map = {
    "get_ticker": get_ticker,
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_daily_stock_performance": get_daily_stock_performance,
}


functions = [
    {
        "type": "function",
        "function": {
            "name": "get_ticker",
            "description": "Given the name of a company returns its ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The name of the company",
                    }
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_statement",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's income statement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance_sheet",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's balance sheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_stock_performance",
            "description": "Given a ticker symbol (i.e AAPL) returns the performance of the stock for the last 100 days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    for message in messages:
        with st.chat_message(message.role):
            st.markdown(message.content[0].text.value.replace("$", "\$"))


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id, thread_id=thread_id, tool_outputs=outputs
    )


#! api키 받기
if "api" not in st.session_state:
    st.session_state["api"] = ""

if not st.session_state["api"]:
    with st.sidebar:
        st.session_state["api"] = st.text_input("Write your openAI API Key.")
        st.button("Accept")
else:
    with st.sidebar:
        st.write(f'API-KEY: {st.session_state["api"]}')
        st.write(
            "REPO: https://github.com/DI-Kim/fullstack-gpt/commit/4f3374e90717311a892f00fbfa3a545a1d45b234"
        )
    client.api_key = st.session_state["api"]

    if "assistant_id" not in st.session_state:
        st.session_state["assistant_id"] = ""
    if "thread" not in st.session_state:
        st.session_state["thread"] = None
    #! 어시스턴트 생성 전
    if st.session_state["assistant_id"] == "":
        assistant = client.beta.assistants.create(
            name="Investor Assistant",
            instructions="You help users do research on publicly traded companies and you help users decide if they should buy the stock or not.",
            model="gpt-4-1106-preview",
            tools=functions,
        )
        # ! 생성 후 id 저장
        st.session_state["assistant_id"] = assistant.id
        # st.session_state["assistant_id"] = "asst_sNRfGqBlfUQnarB6j5clyGxW"
    st.markdown(
        """
    WELCOME
              
              Use this chatbot to ask for information about the company you are curious about!
    """
    )
    thread = None
    if st.session_state["thread"] is not None:
        content = st.chat_input("Write...")
        send_message(st.session_state["thread"].id, content)
    else:
        content = st.chat_input("Write...")
        #! 스레드 생성
        if content:
            st.session_state["thread"] = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            )

    if content:
        #! 런 생성
        run = client.beta.threads.runs.create(
            thread_id=st.session_state["thread"].id,
            assistant_id=st.session_state["assistant_id"],
        )

        while get_run(run.id, st.session_state["thread"].id).status != "completed":
            with st.spinner("Loading..."):
                while (
                    get_run(run.id, st.session_state["thread"].id).status
                    == "in_progress"
                ):
                    time.sleep(1)
            st.success(get_run(run.id, st.session_state["thread"].id).status)
            if (
                get_run(run.id, st.session_state["thread"].id).status
                == "requires_action"
            ):
                submit_tool_outputs(run.id, st.session_state["thread"].id)

        get_messages(st.session_state["thread"].id)
