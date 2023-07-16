"""
Treina um modelo PyTorch de classificação de imagens
usando código agnóstico a device
"""

import os
from timeit import default_timer as timer
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils

#Configura hyperparâmetros
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

#Configura pastas
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

#Configura código agnóstico a device
device = "cuda" if torch.cuda.is_available() else "cpu"

#Cria transforms
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

#Cria DataLoaders e  class_names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir, test_dir, data_transform, BATCH_SIZE
)

#Cria o modelo
model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=HIDDEN_UNITS,
                              output_shape=len(class_names)
                              ).to(device)

#Define Loss e Otimizador
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start the timer
start_time = timer()

#Começa o treinamento
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             optimizer=optimizer,
             loss_fn=loss_fn,
             epochs=NUM_EPOCHS,
             device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
