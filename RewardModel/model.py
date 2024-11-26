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
        print("good_input_ids", good_input_ids)
        bad_input_ids = kwargs.get("bad_input_ids")
        print("bad", bad_input_ids)
        good_attention_mask = kwargs.get("good_attention_mask")
        bad_attention_mask = kwargs.get("bad_attention_mask")
        
        if bad_input_ids is not None:

            good_roberta = self.roberta(good_input_ids, attention_mask=good_attention_mask).last_hidden_state

            print("good roberta", good_roberta)

            chosen_reward = self.classifier(good_roberta).squeeze(-1)

            print("chosen_reward", chosen_reward)

            bad_roberta = self.roberta(bad_input_ids, attention_mask=bad_attention_mask).last_hidden_state

            print("bad roberta", bad_roberta)

            rejected_reward = self.classifier(bad_roberta).squeeze(-1)

            print("rejected_reward", rejected_reward)

            print("cls token chosen", chosen_reward[:,0].squeeze(-1))

            print("cls token rejected", rejected_reward[:,0].squeeze(-1))
            return chosen_reward[:,0].squeeze(-1), rejected_reward[:,0].squeeze(-1)

        roberta = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        print("normal inputids", input_ids)
        print("normal roberta", roberta)

        reward = self.classifier(roberta).squeeze(-1)
        print("normal reward", reward)
        return reward[:,0].squeeze(-1)

