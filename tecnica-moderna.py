import re
import pandas as pd
from datasets import load_dataset
from collections import deque


ds = load_dataset("openai/graphwalks")
ds = ds.with_format("pandas")

df = ds["train"].to_pandas()


def parse_prompt(row):
    prompt = row["prompt"]
    problem_type = row["problem_type"]

    main_text = prompt.split('<end example>')[1]

    edges_match = re.search(
        r"The graph has the following edges:\n(.*?)\n\nOperation:",
        main_text,
        re.DOTALL
    )

    if problem_type == "bfs":
        node_match = re.search(r"Perform a BFS from node ([a-zA-Z0-9]+) with depth", main_text)
        depth_match = re.search(r"with depth (\d+)", main_text)
        node = node_match.group(1) if node_match else None
        depth = int(depth_match.group(1)) if depth_match else None

    elif problem_type == "parents":
        node_match = re.search(r"Find the parents of node ([a-zA-Z0-9]+)\.", main_text)
        node = node_match.group(1) if node_match else None
        depth = None

    else:
        node = None
        depth = None

    return {
        "edges": edges_match.group(1).strip() if edges_match else None,
        "target_node": node,
        "depth": depth
    }

# Construção do grafo
def constroi_grafo(lista_arestas, grafo=None):
    if grafo is None:
        grafo = {}

    for aresta in lista_arestas:
        origem, destino = aresta.split("->")
        origem = origem.strip()
        destino = destino.strip()

        if origem not in grafo:
            grafo[origem] = []

        grafo[origem].append(destino)

    return grafo

# Achar pais de um nó alvo
def pais(no_alvo, grafo):
    lista_pais = []

    for origem, destinos in grafo.items():
        if no_alvo in destinos:
            lista_pais.append(origem)

    return lista_pais

# Achar nós que saem de nó alvo com profundidade p
def busca_largura(no_alvo, grafo, profundidade_max):
    fila = deque([(no_alvo, 0)])
    visitados = set()
    resultado = []

    while fila:
        no_atual, profundidade_atual = fila.popleft()

        if profundidade_atual > profundidade_max:
            continue

        if no_atual in visitados:
            continue

        visitados.add(no_atual)

        if profundidade_atual == profundidade_max:
            resultado.append(no_atual)
        else:
            for vizinho in grafo.get(no_atual, []):
                fila.append((vizinho, profundidade_atual + 1))

    return resultado

# Funcao principal
def busca_grafo(lista_arestas,no_alvo,tipo,profundidade):
    grafo = constroi_grafo(lista_arestas)
    if tipo == "parents":
        return pais(no_alvo, grafo)
    elif tipo == "bfs":
        return busca_largura(no_alvo, grafo,profundidade)
    else:
        print("Erro")
        return None

def solve_problem(row):
    parsed_prompt = parse_prompt(row)

    lista_arestas = parsed_prompt["edges"].split("\n")
    no_alvo = parsed_prompt["target_node"]
    tipo_problema = row["problem_type"]
    profundidade = parsed_prompt["depth"]

    resultado = busca_grafo(lista_arestas, no_alvo, tipo_problema, profundidade)
    
    if set(resultado) == set(row["answer_nodes"]):
        print("Acertou")
    else:
        print("Errou")
    

# Filter 50 examples of each problem type
df_parents = df[df["problem_type"] == "parents"].sample(n=50, random_state=42)
df_bfs = df[df["problem_type"] == "bfs"].sample(n=50, random_state=42)

#df_subset = pd.concat([df_parents, df_bfs]).reset_index(drop=True)
#df_bfs.apply(solve_problem, axis=1)

df.apply(solve_problem, axis=1)