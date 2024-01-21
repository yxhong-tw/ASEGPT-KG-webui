import re
from typing import List, Tuple

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
            triplets.append((subject, relation, obj))

    return triplets
