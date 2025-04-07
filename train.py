from model import RegressionController
import lightning as L
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

# ~ HYPERPARAMETERS ~
INPUT_SIZE = 2
HIDDEN_SIZE = 3
OUTPUT_SIZE = 12
MAX_ITERATIONS = 200
BATCH_SIZE = 2

# ~ TRAINING ~
data = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])

print(f"Data: {data.shape}")

labels = torch.tensor([[0.0000, 0.0169, 0.0339, 0.0508, 0.0678, 0.0847, 0.1017, 0.1186, 0.1356,
         0.1525, 0.1695, 0.1864],
        [0.2034, 0.2203, 0.2373, 0.2542, 0.2712, 0.2881, 0.3051, 0.3220, 0.3390,
         0.3559, 0.3729, 0.3898],
        [0.4068, 0.4237, 0.4407, 0.4576, 0.4746, 0.4915, 0.5085, 0.5254, 0.5424,
         0.5593, 0.5763, 0.5932],
        [0.6102, 0.6271, 0.6441, 0.6610, 0.6780, 0.6949, 0.7119, 0.7288, 0.7458,
         0.7627, 0.7797, 0.7966],
        [0.8136, 0.8305, 0.8475, 0.8644, 0.8814, 0.8983, 0.9153, 0.9322, 0.9492,
         0.9661, 0.9831, 1.0000]])

print(f"Control Values: {labels.shape}")

dataset = torch.utils.data.TensorDataset(data, labels)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
model = RegressionController(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)

trainer = L.Trainer(max_steps=MAX_ITERATIONS)
trainer.fit(model=model, train_dataloaders=train_loader)

print(f"Training loss: {model.get_final_loss()}")