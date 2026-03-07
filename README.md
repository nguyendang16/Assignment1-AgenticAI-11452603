# Financial Assistant CLI

A command-line chatbot that answers questions about **exchange rates** and **stock prices** using an LLM with tool calling. Implemented without high-level frameworks (e.g. LangChain).

## Features

- **Tools:** `get_exchange_rate` (USD_TWD, JPY_TWD, EUR_USD) and `get_stock_price` (AAPL, TSLA, NVDA)
- **Mock data only** — no live API calls; data is fixed for grading consistency
- **Function map** for tool dispatch; **parallel tool calls** in one turn
- **Context-aware** — remembers earlier messages (e.g. “What is its price?”)

## Setup

1. **Clone or download** this project.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **API key:** Create a `.env` file in the project root with your OpenAI (or compatible) API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```
   Do not commit `.env` or upload keys.

## Run

```bash
python main.py
```

Type your questions; type `exit` or `quit` to end.

## Example prompts

- *What is the USD to TWD exchange rate?*
- *Price of AAPL and TSLA?*
- *What about EUR_USD?*

## Requirements

- Python 3.8+
- OpenAI Python SDK (or compatible API that supports tool/function calling)
- Model: e.g. `gpt-4o-mini` (or any OpenAI-compatible model with tool use)
