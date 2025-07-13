import heapq
import re
from collections import defaultdict
import pandas as pd
from datasets import load_dataset

output_esperado = open("./outputs/tecnica classica/output esperado.txt", "w", encoding="utf-8")
output_busca_a_estrela = open("./outputs/tecnica classica/output busca A-estrela.txt", "w", encoding="utf-8")

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

def constroi_grafo(lista_arestas, grafo=None):
    if grafo is None:
        grafo = defaultdict(list)

    for aresta in lista_arestas:
        origem, destino = aresta.split("->")
        origem = origem.strip()
        destino = destino.strip()

        grafo[origem].append(destino)

    return grafo


def constroi_grafo_reverso(lista_arestas):
    grafo_reverso = defaultdict(list)
    grau_entrada = defaultdict(int)

    for aresta in lista_arestas:
        origem, destino = aresta.split("->")
        origem = origem.strip()
        destino = destino.strip()

        grafo_reverso[destino].append(origem)
        grau_entrada[destino] += 1  # conta quantos chegam no destino

    return grafo_reverso, grau_entrada


def heuristica_por_grau(n, grau_entrada):
    return 1 / (1 + grau_entrada.get(n, 0))  # mais conexões, menor h(n)

def busca_a_estrela_parents(grafo_reverso, grau_entrada, no_alvo):
    fila = []
    heapq.heappush(fila, (0, 0, no_alvo, []))  # f, g, nó atual, caminho
    visitados = {}
    predecessores = defaultdict(list)

    menor_custo = None

    while fila:
        f, g, atual, caminho = heapq.heappop(fila)

        if atual in visitados and visitados[atual] <= g:
            continue

        visitados[atual] = g
        caminho_atual = caminho + [atual]

        for vizinho in grafo_reverso.get(atual, []):
            novo_g = g + 1
            h = heuristica_por_grau(vizinho, grau_entrada)
            f_score = novo_g + h
            heapq.heappush(fila, (f_score, novo_g, vizinho, caminho_atual))

            # se é a primeira vez ou tem menor custo, atualiza
            if menor_custo is None or novo_g < menor_custo:
                menor_custo = novo_g
                predecessores = defaultdict(list)
                predecessores[novo_g].append(vizinho)
            elif novo_g == menor_custo:
                predecessores[novo_g].append(vizinho)

    return predecessores.get(menor_custo, [])


def busca_a_estrela_bfs(no_inicial, grafo, profundidade_max):
    # fila de prioridade: (f(n), g(n), no_atual)
    # f(n) = g(n) + h(n), onde:
    #   g(n) = profundidade atual
    #   h(n) = profundidade_max - profundidade atual
    fila = [(0 + (profundidade_max - 0), 0, no_inicial)]
    visitados = {}
    resultado = set()

    while fila:
        f, g, no_atual = heapq.heappop(fila)

        # ignorar se já visitamos este nó com menor custo
        if no_atual in visitados and visitados[no_atual] <= g:
            continue

        visitados[no_atual] = g

        if g == profundidade_max:
            resultado.add(no_atual)
            continue

        for vizinho in grafo.get(no_atual, []):
            novo_g = g + 1
            h = profundidade_max - novo_g
            f = novo_g + h
            heapq.heappush(fila, (f, novo_g, vizinho))

    return list(resultado)

def busca_grafo(lista_arestas, no_alvo, tipo, profundidade=None):
    if tipo == "parents":
        grafo, graus = constroi_grafo_reverso(lista_arestas)
        return busca_a_estrela_parents(grafo, graus, no_alvo)
    elif tipo == "bfs":
        grafo = constroi_grafo(lista_arestas)
        return busca_a_estrela_bfs(no_alvo, grafo, profundidade)
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

    output_esperado.write(str(row["answer_nodes"]))
    output_esperado.write("\n")

    output_busca_a_estrela.write(str(resultado))
    output_busca_a_estrela.write("\n")

# filtrando problemas exemplo
df_parents = df[(df["problem_type"] == "parents") & (df["prompt_chars"] <= 20000)].head(115)
df_bfs = df[(df["problem_type"] == "bfs") & (df["prompt_chars"] <= 20000)].head(115)

df_subset = pd.concat([df_parents, df_bfs]).reset_index(drop=True)
df_subset.apply(solve_problem, axis=1)

print("Fim da Execução")