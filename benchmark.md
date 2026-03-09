Task A (Persona): Input "Who are you?" -> Show it replies as a Financial Assistant.

Task B (Single Tool): Input "What is the price of NVDA?" -> Show it returns 190.00 .

Task C (Parallel Tools): Input "Compare the stock prices of AAPL and TSLA." -> Show the debug log executing both tools in one turn, and the final answer comparing 260.00 vs 430.00 .

Task D (Memory Test):

Step 1: Input "My name is [Your Name]." -> Agent acknowledges.

Step 2: Input "What is my name?" -> Agent correctly retrieves the name from memory.

Task E (Error Handling): Input "What is the price of GOOG?" -> Show it handles the unknown data gracefully (eg, "Data not found") without crashing.