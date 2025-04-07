import torch 
import lightning as L
import torch.nn.functional as F
import torch.nn as nn   


class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class RegressionController(L.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = MLPRegressor(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        self.save_hyperparameters()
        self.final_loss = None
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        self.final_loss = loss  # Store the final loss
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        return optimizer
    
    def get_final_loss(self):
        return self.final_loss.item() if self.final_loss is not None else None