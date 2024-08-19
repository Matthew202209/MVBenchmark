import torch
from torch import nn
from transformers import AutoModelForMaskedLM, AutoConfig



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
