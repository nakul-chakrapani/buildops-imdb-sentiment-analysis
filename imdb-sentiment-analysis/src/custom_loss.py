'''
Create all custom loss functions here
'''
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, parameter=None):
        super().__init__()
        self.parameter = parameter

    def forward(self, pred_logits, target, num_tokens):
        loss_fn = nn.BCEWithLogitsLoss()
        pred_logits = pred_logits.view(-1)
        target = target.float()
        loss = loss_fn(pred_logits, target)

        # Simple Custom loss logic
        # Intuition behind is Longer the length of the review(in terms of words/tokens) more information is given by the reviewer
        # Also from the data analysis before, on average a review will have 225+ tokens
        # This loss function penalizes wrong predictions on longer reviews

        if self.parameter is not None:
            loss += torch.sum(self.parameter * (num_tokens)/512)
        return loss
