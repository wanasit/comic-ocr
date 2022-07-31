from typing import List, Optional

CHAR_ID_PADDING = 0
CHAR_ID_UNKNOWN = 1
SUPPORT_CHARACTERS = ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?"#$%&\'`()*+,-.'
SUPPORT_DICT_SIZE = len(SUPPORT_CHARACTERS) + 2
CHAR_TO_ID = {c: i + 2 for i, c in enumerate(SUPPORT_CHARACTERS)}
ID_TO_CHAR = {i + 2: c for i, c in enumerate(SUPPORT_CHARACTERS)}


def encode(text: str, padded_output_size: Optional[int] = None) -> List[int]:
    encoded = [CHAR_TO_ID[c] if c in CHAR_TO_ID else CHAR_ID_UNKNOWN for c in text]
    if padded_output_size and len(encoded) < padded_output_size:
        encoded = encoded + [0] * (padded_output_size - len(encoded))
    return encoded


def decode(prediction: List[int], unknown_char='?') -> str:
    predicted_id_chunks = [[]]
    for id in prediction:
        if id == 0:
            predicted_id_chunks.append([])
            continue
        if predicted_id_chunks[-1] and predicted_id_chunks[-1][-1] == id:
            continue
        predicted_id_chunks[-1].append(id)
    text = ''
    for chunk in predicted_id_chunks:
        for id in chunk:
            if id in ID_TO_CHAR:
                text += ID_TO_CHAR[id]
            else:
                text += unknown_char
    return text