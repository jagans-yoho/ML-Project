import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

texts = [
    "Hello world!",
    "Customer support automation using natural language processing.",
    (
        "Artificial Intelligence engineers must balance precision, recall,"
        " latency, and compute cost."
    ),
    (
        "Supercalifragilisticexpialidocious is an extremely complex"
        " morphological compound token."
    ),
]

print(" Token Counts & Words Per Token ")
for t in texts:
  token_count = len(enc.encode(t))
  word_count = len(t.split())
  wpt = word_count / token_count
  print(
      f"Text: '{t[:40]}...' | Words: {word_count} | Tokens: {token_count} |"
      f" Words/Token: {wpt:.2f}"
  )

daily_input = 300 * 1200  
daily_output = 300 * 300  
monthly_input = daily_input * 30  
monthly_output = daily_output * 30  

cost_input = (monthly_input / 1_000_000) * 2.50
cost_output = (monthly_output / 1_000_000) * 10.00
total_monthly_cost = cost_input + cost_output

print(f"\nEstimated Monthly Cost: ${total_monthly_cost:.2f}")