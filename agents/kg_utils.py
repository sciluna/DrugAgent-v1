import math
import pickle
import traceback
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from cache import FileCache


class KnowledgeGraph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, src, dest):
        self.graph[src].append(dest)
        self.graph[dest].append(src)

    def get_all_paths(
        self, start: str, end: str, max_length: int = 4, max_paths: int = 5
    ) -> List[List[str]]:

        def dfs(current, target, path, paths, visited):
            if len(paths) >= max_paths:
                return
            if (len(path) - 1) > max_length:
                return
            if current == target:
                paths.append(path[:])
                return

            for neighbor in self.graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dfs(neighbor, target, path + [neighbor], paths, visited)
                    visited.remove(neighbor)

        paths: list[list[str]] = []
        visited: set[str] = {start}
        dfs(start, end, [start], paths, visited)
        return paths

    def shortest_path(self, start, end):
        if start == end:
            return {"distance": 0, "path": [start]}
        visited = set()
        queue = deque([(start, 0)])
        parent = {}

        while queue:
            node, distance = queue.popleft()

            if node == end:
                path = []
                current = end
                while current is not None:
                    path.append(current)
                    current = parent.get(current)
                path.reverse()
                return {"distance": distance, "path": path}

            if node not in visited:
                visited.add(node)

                for neighbor in self.graph[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))
                        parent[neighbor] = node

        return {"distance": -1, "path": None}

    def get_neighbors(self, node):
        return self.graph[node]

    def get_all_nodes(self):
        return list(self.graph.keys())


def load_kg(file_path) -> KnowledgeGraph:
    with open(file_path, "rb") as f:
        kg = pickle.load(f)
    return kg


kg = load_kg("knowledge_graph.pkl")

dti_score_cache = FileCache("kg_dti_scores")


_DEG: Dict[str, int] = {n: len(neigh) for n, neigh in kg.graph.items()}
_NUM_NODES: int = len(_DEG)


def _path_weight(path: List[str]) -> float:
    hops = len(path) - 1
    if hops <= 0:
        return 0.0

    s = 0.0
    for i in range(hops):
        n_i = path[i]
        n_j = path[i + 1]
        s += (_DEG.get(n_i, 0) + _DEG.get(n_j, 0)) / (2.0 * _NUM_NODES)

    return (1.0 / hops) * s


def _raw_kg_score_and_path(
    drug: str,
    target: str,
    max_hops: int = 4,
    max_paths: int = 5,
    scaling_factor: float = 1.0,  # set to 10.0 if you keep the constant in the paper
) -> Tuple[float, Optional[List[str]]]:
    paths = kg.get_all_paths(drug, target, max_length=max_hops, max_paths=max_paths)
    if not paths:
        return 0.0, None

    best_score = 0.0
    best_path: Optional[List[str]] = None

    for p in paths:
        hops = len(p) - 1
        if hops <= 0:
            continue
        w = _path_weight(p)
        score = (scaling_factor * w) / math.log1p(hops)  # ln(1+|p|)
        if score > best_score:
            best_score = score
            best_path = p

    return best_score, best_path


def calculate_dti_score(drug, target):
    try:
        if drug not in kg.graph:
            reasoning = f"reasoning: Drug {drug} not found in knowledge graph."
            return 0.0, reasoning

        if target not in kg.graph:
            reasoning = f"reasoning: Target {target} not found in knowledge graph."
            return 0.0, reasoning

        # IMPORTANT: set scaling_factor=10.0 ONLY if your manuscript keeps the constant 10
        score, best_path = _raw_kg_score_and_path(
            drug, target, max_hops=4, max_paths=5, scaling_factor=1.0
        )

        if best_path is None or score <= 0.0:
            reasoning = (
                f"reasoning: No valid path found between {drug} and {target} "
                f"within 4 hops (searched up to 5 paths)."
            )
            return 0.0, reasoning

        hops = len(best_path) - 1
        w = _path_weight(best_path)
        reasoning = (
            f"reasoning: Selected highest-scoring path between {drug} and {target} "
            f"(hops={hops}). Computed w(p)={w:.6g} and score={score:.6g} "
            f"using w(p)/ln(1+|p|). Path is {best_path}."
        )
        return float(score), reasoning

    except Exception as e:
        print(
            f"Error calculating DTI score for drug '{drug}' and target '{target}': {e}"
        )
        print(traceback.format_exc())
        return 0.0, "reasoning: Error occurred while calculating DTI score."


def get_dti_score(drug: str, target: str) -> tuple[float, str]:
    try:
        cache_key = f"{drug}_{target}"
        cached_score = dti_score_cache.get(cache_key)
        if cached_score is not None:
            return cached_score["score"], cached_score.get(
                "reasoning", "reasoning: Cached result."
            )
        else:
            score, reasoning = calculate_dti_score(drug, target)
            dti_score_cache.set(cache_key, {"score": score, "reasoning": reasoning})
            return score, reasoning
    except Exception as e:
        print(f"Error getting DTI score for drug '{drug}' and target '{target}': {e}")
        print(traceback.format_exc())
        return 0, "reasoning: Error occurred while retrieving DTI score."


def get_dti_scores(
    drugs: List[str], targets: List[str]
) -> List[List[Union[str, float, str]]]:
    scores: List[List[Union[str, float, str]]] = []
    for drug, target in zip(drugs, targets):
        try:
            score, reasoning = get_dti_score(drug, target)
            scores.append([drug, target, score, reasoning])
        except Exception as e:
            print(f"Error processing drug '{drug}' and target '{target}': {e}")
            print(traceback.format_exc())
            scores.append(
                [
                    drug,
                    target,
                    0,
                    "reasoning: Error occurred during score calculation.",
                ]
            )
    print("kg scores: ", scores)
    return scores
