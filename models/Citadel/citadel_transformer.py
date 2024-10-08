from torch import nn
from transformers import AutoTokenizer

class HFTransform(nn.Module):
    def __init__(
        self,
        model_path: str = "roberta-base",
        max_seq_len: int = 512,
        add_special_tokens: bool = True,
        return_tensors: bool = True,
    ):
        super().__init__()
        # remove recursive argument which is not supported now
        # local_model_path = PathManager.get_local_path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sep_token = self.tokenizer.sep_token
        self.max_seq_len = max_seq_len
        self.add_special_tokens = add_special_tokens
        self.return_tensors = return_tensors

    def forward(self, texts, text_pair=None, padding=True):

        result =self.tokenizer(
            texts,
            text_pair,
            return_tensors="pt" if self.return_tensors else None,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=self.add_special_tokens)

        result.data =  {k: v.squeeze(0) for k, v in result.data.items()}
        return result

