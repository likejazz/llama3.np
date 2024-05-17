from typing import List
import json


class Tokenizer:
    def __init__(self, model_path: str):
        with open(model_path, "r", encoding="utf-8") as f:
            model = json.load(f)
        self.vocab = model["tokens"]
        self.scores = model["scores"]
        self.bos_id = 1
        self.eos_id = 2

    def str_lookup(self, token: str) -> int:
        try:
            index = self.vocab.index(token)
            return index
        except ValueError as err:
            return -1

    def encode(
            self,
            text: str,
            add_bos: bool = True,
            add_eos: bool = False,
    ) -> List[int]:
        tokens = []
        for pos, char in enumerate(text):
            id = self.str_lookup(char)
            if id >= 0:
                tokens.append(id)
        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1

            for i in range(len(tokens) - 1):
                # Check if we can merge the pair (tokens[i], tokens[i+1])
                string = self.vocab[tokens[i]] + self.vocab[tokens[i + 1]]
                id = self.str_lookup(string)
                if id != -1 and self.scores[id] > best_score:
                    best_score = self.scores[id]
                    best_id = id
                    best_idx = i

            if best_idx == -1:
                break

            # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id
            # Delete token at position best_idx+1, shift the entire sequence back 1
            tokens = tokens[0: best_idx + 1] + tokens[best_idx + 2:]
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, ids: List[int]) -> str:
        res = []
        for i in ids:
            token = self.vocab[i]
            res.append(token)
        text = "".join(res)
        text = text.strip("<s>").strip("</s>")
        return text
