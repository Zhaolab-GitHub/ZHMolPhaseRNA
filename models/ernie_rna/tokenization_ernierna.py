import json
import os
from transformers import PreTrainedTokenizer


_VOCAB = {
    "<cls>": 0,
    "<pad>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "G": 4,
    "A": 5,
    "U": 6,
    "C": 7,
    "N": 8,
    "Y": 9,
    "R": 10,
    "S": 11,
    "K": 12,
    "W": 13,
    "M": 14,
    "D": 15,
    "H": 16,
    "V": 17,
    "B": 18,
    "X": 19,
    "I": 20,
    "madeupword0000": 21,
    "madeupword0001": 22,
    "madeupword0002": 23,
    "<mask>": 24,
}


class ErnieRNATokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        cls_token="<cls>",
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        mask_token="<mask>",
        **kwargs,
    ):
        if vocab_file is not None and os.path.isfile(vocab_file):
            with open(vocab_file) as f:
                self._vocab = json.load(f)
        else:
            self._vocab = dict(_VOCAB)
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}

        super().__init__(
            cls_token=cls_token,
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text):
        tokens = []
        for ch in text.upper():
            if ch == "T":
                tokens.append("U")
            elif ch in self._vocab:
                tokens.append(ch)
            else:
                tokens.append("<unk>")
        return tokens

    def _convert_token_to_id(self, token):
        return self._vocab.get(token, self._vocab["<unk>"])

    def _convert_id_to_token(self, index):
        return self._ids_to_tokens.get(index, "<unk>")

    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)
        fname = (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        path = os.path.join(save_directory, fname)
        with open(path, "w") as f:
            json.dump(self._vocab, f, indent=2)
        return (path,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        cls = [self.cls_token_id]
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + eos
        return cls + token_ids_0 + eos + cls + token_ids_1 + eos

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens=True)
        mask = [1] + [0] * len(token_ids_0) + [1]
        if token_ids_1 is not None:
            mask += [1] + [0] * len(token_ids_1) + [1]
        return mask

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 2)
        return (
            [0] * (len(token_ids_0) + 2)
            + [1] * (len(token_ids_1) + 2)
        )
