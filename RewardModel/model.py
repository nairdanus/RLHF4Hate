import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer

class RoBERTaRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.classifier = torch.nn.Linear(1024, 1)
        


    def forward(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        
        good_input_ids = kwargs.get("good_input_ids")
        bad_input_ids = kwargs.get("bad_input_ids")
        good_attention_mask = kwargs.get("good_attention_mask")
        bad_attention_mask = kwargs.get("bad_attention_mask")

        if bad_input_ids is not None:
            good_roberta = self.roberta(good_input_ids, attention_mask=good_attention_mask).last_hidden_state
            chosen_reward = self.classifier(good_roberta)

            bad_roberta = self.roberta(bad_input_ids, attention_mask=bad_attention_mask).last_hidden_state
            rejected_reward = self.classifier(bad_roberta)
            
            return chosen_reward.mean(), rejected_reward.mean()

        roberta = self.roberta(input_ids, attention_mask=attention_mask).last_hidden_state
        reward = self.classifier(roberta)

        return reward.mean()
