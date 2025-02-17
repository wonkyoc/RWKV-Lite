from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, \
        GPTNeoXForCausalLM, LlamaTokenizer
import torch
import time


with open("models.txt", "r") as f:
    data = f.read().split("\n")
    test_models = [row for row in data if "#" not in row]

test_prompt = "Alice was so tired when she got back home so she went"
token_limit = 200

# download all models
#for m in models_01b + models_04b + models_1b5:
for m in test_models:
    model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float16)

    if "llama" in m.lower():
        tokenizer = LlamaTokenizer.from_pretrained(m)
    else:
        tokenizer = AutoTokenizer.from_pretrained(m)
    input_ids = tokenizer(test_prompt, return_tensors="pt")

    # Generate completion
    exec_time = 0
    iterations = 5
    for _ in range(iterations):
        gen_start_t = time.time()
        output = model.generate(**input_ids, max_length = token_limit)

        # Decode the completion
        output_text = tokenizer.decode(output[0])
        gen_end_t = time.time()

        assert len(output[0]) == token_limit

        exec_time += gen_end_t - gen_start_t
    #print(output_text)
    print(f"model={m} exec_time={exec_time/iterations:.2f} tps={token_limit/(exec_time/iterations):.2f}")
