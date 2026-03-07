"""
Financial Assistant CLI — LLM Agent with Function Map and Parallel Tool Calls.
Uses mock data for exchange rates and stock prices. No high-level agent frameworks.
"""
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. Setup & Security — API key from environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. Mock Data 

EXCHANGE_RATES = {
    "USD_TWD": "32.0",
    "JPY_TWD": "0.2",
    "EUR_USD": "1.2",
}

STOCK_PRICES = {
    "AAPL": "260.00",
    "TSLA": "430.00",
    "NVDA": "190.00",
}


def get_exchange_rate(currency_pair: str) -> str:
    """Get exchange rate for a currency pair. Returns JSON string."""
    if currency_pair in EXCHANGE_RATES:
        return json.dumps({
            "currency_pair": currency_pair,
            "rate": EXCHANGE_RATES[currency_pair],
        })
    return json.dumps({"error": "Data not found"})


def get_stock_price(symbol: str) -> str:
    """Get stock price for a symbol. Returns JSON string."""
    if symbol.upper() in STOCK_PRICES:
        return json.dumps({
            "symbol": symbol.upper(),
            "price": STOCK_PRICES[symbol.upper()],
        })
    return json.dumps({"error": "Data not found"})


# 3. Function Map — dispatch tool calls without if-else chains
available_functions = {
    "get_exchange_rate": get_exchange_rate,
    "get_stock_price": get_stock_price,
}

# 4. Tool Schemas 
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get the exchange rate for a currency pair. Supported pairs: USD_TWD, JPY_TWD, EUR_USD.",
            "parameters": {
                "type": "object",
                "properties": {
                    "currency_pair": {
                        "type": "string",
                        "description": "The currency pair, e.g. USD_TWD, JPY_TWD, EUR_USD",
                    }
                },
                "required": ["currency_pair"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a symbol. Supported symbols: AAPL, TSLA, NVDA.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. AAPL, TSLA, NVDA",
                    }
                },
                "required": ["symbol"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


def run_agent():
    # 5. System prompt (Financial Assistant persona)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Financial Assistant. You answer questions about exchange rates "
                "(USD_TWD, JPY_TWD, EUR_USD) and stock prices (AAPL, TSLA, NVDA). "
                "Use the provided tools when the user asks for rates or prices. "
                "Remember context from earlier in the conversation (e.g. 'its price' refers to the last stock discussed)."
            ),
        }
    ]

    print("Financial Assistant started. Type 'exit' or 'quit' to end.\n")

    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        response_msg = response.choices[0].message
        tool_calls = response_msg.tool_calls if response_msg.tool_calls else []

        if tool_calls:
            # Append assistant message (tool call request) to history
            print(f"[DEBUG] Executing {len(tool_calls)} tool(s) in this turn: {[tc.function.name for tc in tool_calls]}")
            messages.append(response_msg)

            # 6. Parallel tool calls — execute all and append all results before next LLM call
            for tool_call in tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                print(f"[DEBUG] Calling {name}({args}) -> ", end="")
                fn = available_functions.get(name)
                if fn:
                    try:
                        result = fn(**args)
                        print(result)
                    except Exception as e:
                        result = json.dumps({"error": str(e)})
                else:
                    result = json.dumps({"error": "Function not found"})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": result,
                })

            # Single follow-up LLM call for final answer (no tools so it returns text)
            final_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            final_content = (final_response.choices[0].message.content or "").strip()
            print(f"Agent: {final_content}")
            messages.append({"role": "assistant", "content": final_content})
        else:
            content = (response_msg.content or "").strip()
            print(f"Agent: {content}")
            messages.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    run_agent()
