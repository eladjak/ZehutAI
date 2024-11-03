from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if torch.cuda.is_available():
    device = "cuda"
    torch.device('cuda')
else:
    torch.device('cpu')
    device = "cpu"

print('device: ' + device)

model = AutoModelForCausalLM.from_pretrained("dicta-il/dictalm2.0-instruct", torch_dtype=torch.bfloat16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictalm2.0-instruct")

messages = [
    {"role": "user", "content": "איזה רוטב אהוב עליך?"},
    {"role": "assistant", "content": "טוב, אני די מחבב כמה טיפות מיץ לימון סחוט טרי. זה מוסיף בדיוק את הכמות הנכונה של טעם חמצמץ לכל מה שאני מבשל במטבח!"},
    {"role": "user", "content": "האם יש לך מתכונים למיונז?"}
]

encoded = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

generated_ids = model.generate(encoded, max_new_tokens=50, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)

decoded = decoded[0].strip().split("\n")
# <s> [INST] איזה רוטב אהוב עליך? [/INST]
# טוב, אני די מחבב כמה טיפות מיץ לימון סחוט טרי. זה מוסיף בדיוק את הכמות הנכונה של טעם חמצמץ לכל מה שאני מבשל במטבח!</s>  [INST] האם יש לך מתכונים למיונז? [/INST]
# בטח, הנה מתכון בסיסי וקל להכנת מיונז ביתי!
#
# מרכיבים:
# - 2 חלמונים גדולים
# - 1 כף חומץ יין לבן
# (it stopped early because we set max_new_tokens=50)
