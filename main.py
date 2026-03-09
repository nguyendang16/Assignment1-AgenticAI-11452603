import json
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. Setup & Security — API key from environment
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_ID = "gemini-2.5-flash"

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
tool_declarations = types.Tool(function_declarations=[
    {
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
        },
    },
    {
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
        },
    },
])


def run_agent():
    # 5. System prompt (Financial Assistant persona)
    config = types.GenerateContentConfig(
        system_instruction=(
            "You are a Financial Assistant. You answer questions about exchange rates "
            "(USD_TWD, JPY_TWD, EUR_USD) and stock prices (AAPL, TSLA, NVDA). "
            "Use the provided tools when the user asks for rates or prices. "
            "Remember context from earlier in the conversation (e.g. 'its price' refers to the last stock discussed)."
        ),
        tools=[tool_declarations],
    )

    contents: list[types.Content] = []

    print("Financial Assistant started. Type 'exit' or 'quit' to end.\n")

    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            )
        )

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=config,
        )

        response_content = response.candidates[0].content
        function_calls = response.function_calls

        if function_calls:
            print(f"[DEBUG] Executing {len(function_calls)} tool(s) in this turn: {[fc.name for fc in function_calls]}")
            contents.append(response_content)

            # 6. Parallel tool calls — execute all and append all results before next LLM call
            function_response_parts = []
            for fc in function_calls:
                name = fc.name
                args = fc.args or {}
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

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response=json.loads(result),
                    )
                )

            contents.append(
                types.Content(parts=function_response_parts, role="user")
            )

            # Single follow-up LLM call for final answer
            final_response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=config,
            )
            final_text = (final_response.text or "").strip()
            print(f"Agent: {final_text}")
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=final_text)],
                )
            )
        else:
            text = (response.text or "").strip()
            print(f"Agent: {text}")
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=text)],
                )
            )


if __name__ == "__main__":
    run_agent()
