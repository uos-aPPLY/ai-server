import re
from typing import List

def group_consecutive(indices: List[int]) -> List[List[int]]:
    """연속된 숫자 그룹으로 묶기"""
    if not indices:
        return []
    
    indices = sorted(set(indices))
    groups = [[indices[0]]]

    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            groups[-1].append(indices[i])
        else:
            groups.append([indices[i]])
    return groups

def mark_by_sentence_indices(diary: str, target_indices: List[int]) -> str:
    sentence_pattern = r'[^.!?…]+[.!?…]?'
    sentences = re.findall(sentence_pattern, diary.strip())
    sentences = [s.strip() for s in sentences]

    grouped_indices = group_consecutive(target_indices)

    result = []
    idx = 1  # 문장 인덱스는 1부터 시작한다고 가정

    i = 0
    while i < len(sentences):
        matched = False
        for group in grouped_indices:
            if idx == group[0]:
                # 감싸야 할 그룹이면
                group_len = len(group)
                joined = ' '.join(sentences[i:i+group_len])
                result.append(f"@{joined}@")
                i += group_len
                idx += group_len
                matched = True
                break
        if not matched:
            result.append(sentences[i])
            i += 1
            idx += 1

    return ' '.join(result)