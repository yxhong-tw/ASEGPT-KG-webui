import re
from typing import Dict, List, Tuple


def parse_triplets(input_string: str) -> List[Tuple[str, str, str]]:
    parts = re.split(r' (?=-\[)|(?<=\]->)|(?=<-\[)|(?<=\]-) ', input_string)

    triplets = []
    for i in range(1, len(parts), 2):
        relation_match = re.search(r'relationship: (.*?)\}', parts[i])
        relation = relation_match.group(1) if relation_match else None

        if '->' in parts[i]:
            subject_match = re.search(r'{name: (.*?)}', parts[i - 1])
            object_match = re.search(r'{name: (.*?)}', parts[i + 1])
        elif '<-' in parts[i]:
            subject_match = re.search(r'{name: (.*?)}', parts[i + 1])
            object_match = re.search(r'{name: (.*?)}', parts[i - 1])

        subject = subject_match.group(1) if subject_match else None
        obj = object_match.group(1) if object_match else None

        if subject and relation and obj:
            triplets.append(f'{subject}, {relation}, {obj}')

    return triplets


def locate_index(target: str, content: str,
                 words: Dict[str, int]) -> Tuple[List[int], List[int]]:
    if target not in words:
        words[target] = len(words)

    start_offsets = []
    end_offsets = []
    for match in re.finditer(pattern=target, string=content):
        start_offsets.append(match.start())
        end_offsets.append(match.end())

    return start_offsets, end_offsets


def find_triplet_index(
    triplets: List[str], content: str
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, int]]:
    words = {}

    relations = []
    entities = []
    for index, triplet_string in enumerate(triplets):
        triplet = triplet_string.split(', ')
        head, relation, tail = triplet

        # Process head
        try:
            head_start_offsets, head_end_offsets = locate_index(
                target=head, content=content, words=words)
        except:
            print(f"Can not locate index {index}'s head ({head})")
            continue

        if len(head_start_offsets) == 0:
            print(f"Index {index} has no head ({head})")
            continue

        # Process relation
        try:
            relation_start_offsets, relation_end_offsets = locate_index(
                target=relation, content=content, words=words)
        except:
            print(f"Can not locate index {index}'s relation ({relation})")
            continue

        if len(relation_start_offsets) == 0:
            print(f"Index {index} has no relation ({relation})")
            continue

        # Process tail

        try:
            tail_start_offsets, tail_end_offsets = locate_index(
                target=tail, content=content, words=words)
        except:
            print(f"Can not locate index {index}'s tail ({tail})")
            continue

        if len(tail_start_offsets) == 0:
            print(f"Index {index} has no tail ({tail})")
            continue

        best_head_offset_index = None
        best_tail_offset_index = None
        hrt_shortest_distance = None
        for relation_offset_index, relation_start_offset in enumerate(
                relation_start_offsets):
            for head_offset_index, head_start_offset in enumerate(
                    head_start_offsets):
                for tail_offset_index, tail_start_offset in enumerate(
                        tail_start_offsets):
                    hrt_distance = abs(head_start_offset -
                                       relation_start_offset) + abs(
                                           tail_start_offset -
                                           relation_start_offset)
                    if hrt_shortest_distance is None or hrt_distance < hrt_shortest_distance:
                        hrt_shortest_distance = hrt_distance
                        best_head_offset_index = head_offset_index
                        best_tail_offset_index = tail_offset_index

        entities.append({
            'id':
            words[head],
            'start_offset':
            head_start_offsets[best_head_offset_index],
            'end_offset':
            head_end_offsets[best_head_offset_index]
        })
        relations.append({
            'id': words[relation],
            'from_id': words[head],
            'to_id': words[tail]
        })
        entities.append({
            'id':
            words[tail],
            'start_offset':
            tail_start_offsets[best_tail_offset_index],
            'end_offset':
            tail_end_offsets[best_tail_offset_index]
        })

    return relations, entities, words
