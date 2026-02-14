"""Test script for DictaLM 2.0 Hebrew language model.

Demonstrates basic conversational generation using the DictaLM 2.0
instruct model with Hebrew chat prompts.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Detect device
device: str = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    "dicta-il/dictalm2.0-instruct",
    torch_dtype=torch.bfloat16,
    device_map=device,
)
tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictalm2.0-instruct")

messages = [
    {"role": "user", "content": "איזה רוטב אהוב עליך?"},
    {
        "role": "assistant",
        "content": "טוב, אני די מחבב כמה טיפות מיץ לימון סחוט טרי. "
        "זה מוסיף בדיוק את הכמות הנכונה של טעם חמצמץ לכל מה שאני מבשל במטבח!",
    },
    {"role": "user", "content": "האם יש לך מתכונים למיונז?"},
]

encoded = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

generated_ids = model.generate(encoded, max_new_tokens=50, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)

decoded = decoded[0].strip().split("\n")
print("\n".join(decoded))
