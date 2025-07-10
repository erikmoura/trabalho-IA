import re
import pandas as pd
from datasets import load_dataset
from collections import deque
from gpt4all import GPT4All


ds = load_dataset("openai/graphwalks")
ds = ds.with_format("pandas")

df = ds["train"].to_pandas()

model = GPT4All("Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf", device="nvidia", n_ctx=4096)
lista_respostas = []

#print(model.generate(df["prompt"].iloc[0]))

def resposta(row):

    resposta_llm = model.generate(row["prompt"])

    lista_respostas.append(resposta_llm)

df_bfs = df[(df["problem_type"] == "bfs") & (df["prompt_chars"] <= 2700)].sample(n=10)

df_bfs.apply(resposta, axis=1)
print(lista_respostas)

