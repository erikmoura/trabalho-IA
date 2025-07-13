import re
import numpy as np
import evaluate as ev
from sklearn.metrics import accuracy_score

# Técnica Moderna - LLM

with open("./outputs/tecnica moderna/output LLM.txt", 'r', encoding='utf-8') as f:
    texto = f.read()

padrao = r'INICIO DA RESPOSTA:\s*(.*?)\s*FIM DA RESPOSTA:'
respostas_LLM = re.findall(padrao, texto, re.DOTALL)
array_respostas_LLM = [resposta.strip() for resposta in respostas_LLM]

with open("./outputs/tecnica moderna/output esperado.txt", 'r', encoding='utf-8') as f:
    array_respostas_esperadas_LLM = [linha.strip() for linha in f.readlines()]

def exact_match(pred, ref):
    return int(pred.strip().lower() == ref.strip().lower())

def partial_match_LLM(pred, ref):
    resultado = re.search(r"Final Answer: \[[^\]]*\]", pred)
    if not resultado:
        return 0
    resultado = resultado.string.split("Final Answer: ")
    if len(resultado) < 2:
        return 0
    answer_nodes_ref = resultado[1]
    answer_nodes_pred = ref.split("Final answer: ")[1]

    answer_nodes_pred = re.findall(r"([a-f0-9]+)", answer_nodes_pred)
    answer_nodes_ref = re.findall(r"([a-f0-9]+)", answer_nodes_ref)

    return int(set(answer_nodes_ref) == set(answer_nodes_pred))

def partial_match_busca_A_estrela(pred, ref):
    answer_nodes_pred = re.findall(r"([a-f0-9]+)", pred)
    answer_nodes_ref = re.findall(r"([a-f0-9]+)", ref)

    return (set(answer_nodes_pred) == set(answer_nodes_ref))

acc_exata_LLM = sum(exact_match(p, r) for p, r in zip(array_respostas_LLM, array_respostas_esperadas_LLM)) / len(array_respostas_LLM)
acc_parcial_LLM = sum(partial_match_LLM(p, r) for p, r in zip(array_respostas_LLM, array_respostas_esperadas_LLM)) / len(array_respostas_LLM)

# Técnica Clássica - Busca A*

with open("./outputs/tecnica classica/output esperado.txt", 'r', encoding='utf-8') as f:
    array_respostas_esperadas_busca_A_estrela = [linha.strip() for linha in f.readlines()]

with open("./outputs/tecnica classica/output busca A-estrela.txt", 'r', encoding='utf-8') as f:
    array_respostas_busca_A_estrela = [linha.strip() for linha in f.readlines()]

acc_exata_busca_A_estrela = sum(exact_match(p, r) for p, r in zip(array_respostas_busca_A_estrela, array_respostas_esperadas_busca_A_estrela)) / len(array_respostas_busca_A_estrela)
acc_parcial_busca_A_estrela = sum(partial_match_busca_A_estrela(p, r) for p, r in zip(array_respostas_busca_A_estrela, array_respostas_esperadas_busca_A_estrela)) / len(array_respostas_busca_A_estrela)

# Resultados

print("ACURÁCIA:\n")
print("Técnica Clássica (Busca A*):\n")
print(f"Resposta exata: {acc_exata_busca_A_estrela:.2%}")
print(f"Resposta parcial: {acc_parcial_busca_A_estrela:.2%}")
print("\nTécnica Moderna (LLM):\n")
print(f"Resposta exata: {acc_exata_LLM:.2%}")
print(f"Resposta parcial: {acc_parcial_LLM:.2%}")
