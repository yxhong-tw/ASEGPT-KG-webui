from dataclasses import dataclass
from typing import List, Dict


@dataclass
class NodeData:
    topic: str


@dataclass
class Node:
    id: int
    title: str
    label: str
    group: str
    value: float
    data: NodeData


@dataclass
class Edge:
    id: int
    source: int
    target: int
    label: str
    value: float


@dataclass
class PostData:
    nodes: List[Node]
    edges: List[Edge]
    summaries: List[str]
