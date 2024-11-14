import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from .model import RoBERTaRewardModel

class RewardLoss(nn.Module):
    def __init__(self):
        super(RewardLoss, self).__init__()

    def forward(self, chosen_reward, rejected_reward):
        # Calculate the difference and apply sigmoid
        return -torch.log(torch.sigmoid(chosen_reward - rejected_reward))


class SentencePairDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        """
        Args:
            data (list of tuples): Each tuple contains a context, good example, and bad example string.
            tokenizer (RobertaTokenizer): RoBERTa tokenizer for encoding inputs.
            max_length (int): Maximum length of tokenized inputs.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Unpack the tuple containing context, good example, and bad example
        context, good_example, bad_example = self.data[idx]

        # Format as required: "[CLS] Context [SEP] Good example [SEP]" and "[CLS] Context [SEP] Bad example [SEP]"
        good_input = f"<s> {context} </s> {good_example} </s>"
        bad_input = f"<s> {context} </s> {bad_example} </s>"

        # Tokenize each input with RoBERTa's tokenizer
        good_encoding = self.tokenizer(
            good_input, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        bad_encoding = self.tokenizer(
            bad_input, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        # Extract input IDs and attention masks, and squeeze to remove extra dimensions
        good_input_ids = good_encoding['input_ids'].squeeze()
        good_attention_mask = good_encoding['attention_mask'].squeeze()
        
        bad_input_ids = bad_encoding['input_ids'].squeeze()
        bad_attention_mask = bad_encoding['attention_mask'].squeeze()

        return {
            "good_input_ids": good_input_ids,
            "bad_input_ids": bad_input_ids,
            "good_attention_mask": good_attention_mask,
            "bad_attention_mask": bad_attention_mask
        }




class RewardModelTrainer:
    def __init__(self, model, dataloader, epochs, lr):
        self.optimizer = torch.optim.Adam(params =  model.parameters(), lr=lr)
        self.criterion = RewardLoss()
        self.model = model
        self.dataloader = dataloader
        self.epochs = epochs

    def train(self):
        self.model.train()

        best_loss = float('inf')  # Initialize best loss as infinity
        checkpoint_path = "best_model_checkpoint.pth"  # Path to save the best model

        for e in range(self.epochs):
            for batch in self.dataloader:
                self.optimizer.zero_grad()

                # Forward pass through the model
                chosen_reward, rejected_reward = self.model(**batch)

                # Calculate the loss (you can use your custom loss function here)
                loss = self.criterion(chosen_reward, rejected_reward)

                # Backward pass
                loss.backward()

                # Optimize the model
                self.optimizer.step()

                print(f"Epoch {e+1}/{self.epochs}, Loss: {loss.item()}")

                # Save the model checkpoint if we have a new best loss
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    print(f"New best loss: {best_loss}. Saving model checkpoint...")
                    torch.save(self.model.state_dict(), checkpoint_path)

        print(f"Training finished. Best model saved to {checkpoint_path}")
