import pickle
import traceback
from typing import List, Union

import numpy as np
from cache import FileCache
import pickle
import traceback
from typing import List, Union

import numpy as np
from cache import FileCache
from collections import defaultdict, deque

class KnowledgeGraph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, src, dest):
        self.graph[src].append(dest)
        self.graph[dest].append(src)

    def get_all_paths(
        self, start: str, end: str, max_length: int = 4, max_paths: int = 5
    ) -> list:
        """
        Find all paths between start and end nodes within constraints

        Args:
            start: Starting node
            end: Target node
            max_length: Maximum path length
            max_paths: Maximum number of paths to return

        Returns:
            List of paths, where each path is a list of nodes
        """

        def dfs(current, target, path, paths, visited):
            if len(paths) >= max_paths:
                return
            if len(path) > max_length:
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


def calculate_dti_score(drug, target):
    try:
        if drug not in kg.graph:
            reasoning = f"reasoning: Drug {drug} not found in knowledge graph."
            return 0, reasoning

        if target not in kg.graph:
            reasoning = f"reasoning: Target {target} not found in knowledge graph."
            return 0, reasoning

        hops, path = (
            kg.shortest_path(drug, target)["distance"],
            kg.shortest_path(drug, target)["path"],
        )
        if hops == -1:
            reasoning = f"reasoning: No relationship found between {drug} and {target}, path is {path}."
            return 0, reasoning
        elif hops == 1:
            reasoning = f"reasoning: Direct relationship found between {drug} and {target}, path is {path}."
            return 1, reasoning
        else:
            reasoning = f"reasoning: Relationship found between {drug} and {target} with {hops} hops. Calculated score based on logarithmic distance. Path is {path}."
            return 1 / (np.log1p(hops)), reasoning

    except Exception as e:
        print(
            f"Error calculating DTI score for drug '{drug}' and target '{target}': {e}"
        )
        print(traceback.format_exc())
        return 0, "reasoning: Error occurred while calculating DTI score."


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
