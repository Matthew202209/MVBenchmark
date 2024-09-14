import torch
from torch import nn
from transformers import AutoModelForMaskedLM, AutoConfig, AutoTokenizer
import scipy
from models.SLIM.slim_utils import PathManager


class SlimEncoder(nn.Module):
    def __init__(
            self,
            model_path: str = "bert-base-uncased",
            dropout: float = 0.1,
            sparse_mode: bool = False,
    ):
        super().__init__()
        # remove recursive argument which is not supported now
        local_model_path = PathManager.get_local_path(model_path)
        cfg = AutoConfig.from_pretrained(local_model_path)
        cfg.output_hidden_states = True
        cfg.attention_probs_dropout_prob = dropout
        cfg.hidden_dropout_prob = dropout

        self.transformer = AutoModelForMaskedLM.from_pretrained(local_model_path, config=cfg)

    def forward(self, tokens, topk=1):
        ret = {}
        # make it transformer 4.x compatible
        outputs = self.transformer(**tokens, return_dict=True)
        attention_mask = tokens["attention_mask"][:, 1:]
        logits = outputs.logits[:, 1:, :]

        # routing, assign every token to top-k expert
        full_router_repr = torch.log(1 + torch.relu(logits)) * attention_mask.unsqueeze(-1)
        expert_weights, expert_ids = torch.topk(full_router_repr, dim=2, k=topk)  # B x T x topk
        min_expert_weight = torch.min(expert_weights, -1, True)[0]
        sparse_expert_weights = torch.where(full_router_repr >= min_expert_weight, full_router_repr, 0)
        ret["attention_mask"] = attention_mask.clone()
        ret["expert_ids"] = expert_ids.clone()
        ret["expert_weights"] = expert_weights.clone()
        ret["sparse_weights"] = sparse_expert_weights.clone()
        return ret


class Slim_query_encoder:
    def __init__(self, model_name, tokenizer_path, fusion_weight=.99, device='cpu'):
        self.device = "cpu"
        self.fusion_weight = fusion_weight
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.reverse_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        self.weight_range = 5
        self.quant_range = 256

    def encode(self, text, max_length=256, topk=20, return_sparse=False, **kwargs):
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        outputs = self.model(**inputs, return_dict=True)
        attention_mask = inputs["attention_mask"][:, 1:] # remove the cls token
        logits = outputs.logits[:, 1:, :] # remove the cls token prediction
        # routing, assign every token to top-k expert
        full_router_repr = torch.log(1 + torch.relu(logits)) * attention_mask.unsqueeze(-1)
        expert_weights, expert_ids = torch.topk(full_router_repr, dim=2, k=topk) # B x T x topk
        min_expert_weight = torch.min(expert_weights, -1, True)[0]
        sparse_expert_weights = torch.where(full_router_repr >= min_expert_weight, full_router_repr, torch.tensor(0, dtype=full_router_repr.dtype))
        if return_sparse:
            raw_weights, sparse_tok = self._output_to_weight_dicts(expert_weights.cpu(), expert_ids.cpu(), sparse_expert_weights.cpu(), attention_mask.cpu(), return_sparse)[0]
            return self._get_encoded_query_token_wight_dicts([raw_weights])[0], sparse_tok
        else:
            raw_weights = self._output_to_weight_dicts(expert_weights.cpu(), expert_ids.cpu(), sparse_expert_weights.cpu(), attention_mask.cpu(), return_sparse)[0]
            return self._get_encoded_query_token_wight_dicts([raw_weights])[0]

    def _output_to_weight_dicts(self, batch_expert_weights, batch_expert_ids, batch_sparse_expert_weights,
                                batch_attention, return_sparse):
        to_return = []
        for batch_id, sparse_expert_weights in enumerate(batch_sparse_expert_weights):
            tok_vector = scipy.sparse.csr_matrix(sparse_expert_weights.detach().numpy())
            upper_vector, lower_vector = {}, {}
            max_term, max_weight = None, 0
            for position, (expert_topk_ids, expert_topk_weights, attention_score) in enumerate(
                    zip(batch_expert_ids[batch_id],
                        batch_expert_weights[batch_id],
                        batch_attention[batch_id])):
                if attention_score > 0:
                    for expert_id, expert_weight in zip(expert_topk_ids, expert_topk_weights):
                        if expert_weight > 0:
                            term, weight = self.reverse_vocab[expert_id.item()], expert_weight.item()
                            upper_vector[term] = upper_vector.get(term, 0) + weight
                            if weight > max_weight:
                                max_term, max_weight = term, weight
            if max_term is not None:
                lower_vector[term] = lower_vector.get(term, 0) + weight
            fusion_vector = {}
            for term, weight in upper_vector.items():
                fusion_vector[term] = self.fusion_weight * weight + (1 - self.fusion_weight) * lower_vector.get(term, 0)
            if return_sparse:
                to_return.append((fusion_vector, tok_vector))
            else:
                to_return.append(fusion_vector)
        return to_return

    def _get_encoded_query_token_wight_dicts(self, tok_weights):
        to_return = []
        for _tok_weight in tok_weights:
            _weights = {}
            for token, weight in _tok_weight.items():
                weight_quanted = round(weight / self.weight_range * self.quant_range)
                _weights[token] = weight_quanted
            to_return.append(_weights)
        return to_return