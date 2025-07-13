from datasets import load_dataset
from gpt4all import GPT4All

ds = load_dataset("openai/graphwalks")
ds = ds.with_format("pandas")
df = ds["train"].to_pandas()

output_LLM = open("./outputs/tecnica moderna/output LLM.txt", "w", encoding="utf-8")
output_esperado = open("./outputs/tecnica moderna/output esperado.txt", "w", encoding="utf-8")

ind = 1

model = GPT4All("Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf", device="nvidia", n_ctx=12000)

def resposta(row):
    global ind
    print(ind)
    resposta_llm = model.generate(row["prompt"], temp=0.3)
    ans_ini = "INICIO DA RESPOSTA:\n\n"
    output_LLM.write(ans_ini)
    output_LLM.write(resposta_llm)
    ans_end = "\n\nFIM DA RESPOSTA:\n\n" 
    output_LLM.write(ans_end)

    resposta_esperada = "Final answer: " + str(row["answer_nodes"])
    output_esperado.write(resposta_esperada)
    output_esperado.write("\n")
    ind += 1

df_parents = df[(df["problem_type"] == "parents") & (df["prompt_chars"] <= 20000)].head(115)
df_bfs = df[(df["problem_type"] == "bfs") & (df["prompt_chars"] <= 20000)].head(115)

df_parents.apply(resposta, axis=1)
df_bfs.apply(resposta, axis=1)

print("Fim da Execução")