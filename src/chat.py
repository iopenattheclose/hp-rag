from mlx_lm import load, generate
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")

print("Loading Qwen3...")
model, tokenizer = load(MODEL_PATH)
print("Qwen3 is ready! Type 'exit' to quit.\n")

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    }
]

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"exit", "quit"}:
        break

    if not user_input:
        continue

    messages.append({
        "role": "user",
        "content": user_input
    })

    prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=1024,
        verbose=False
    )

    print(f"\nQwen3: {response}\n")

    messages.append({
        "role": "assistant",
        "content": response
    })