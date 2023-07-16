"""
Contém funcionalidades para criar DataLoaders do PyTorch para dados
de classificação de imagens
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str,
                        test_dir: str,
                        transform: transforms.Compose,
                        batch_size: int,
                        num_workers: int=NUM_WORKERS):
  """
  Cria DataLoaders de treino e teste.

  Recebe caminhos de diretórios de treino e teste e os torna em
  Datasets Pytorch e, então, para DataLoaders PyTorch.

  Args:
    train_dir: Caminho para diretório de treino
    test_dir: Caminho para diretório de teste.
    transform: torchvision transforms para serem usados nos dados de treino
      de teste.
    batch_size: Número de instâncias por batch em cada DataLoader
    num_workers: Um inteiro para a quantidade de workers por DataLoader


  Returns:
    Uma tupla de (train_dataloader, test_dataloader, class_names).
    Onde class_names é uma lista das classes target.
    Exemplo:
      train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=path/to/train_dir,
      test_dir=path/to/test_dir,
      transform=some_transform,
      batch_size=32,
      num_workers=4)
  """
  train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

  test_data = datasets.ImageFolder(root=test_dir,
                                 transform=transform)

  train_dataloader = DataLoader(dataset=train_data,
                              batch_size=batch_size, # how many samples per batch?
                              num_workers=num_workers, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True, # shuffle the data?
                              pin_memory=True) #faster data transfer to CUDA-enabled GPUs

  test_dataloader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False,  # don't usually need to shuffle testing data
                             pin_memory=True)

  class_names = train_data.classes

  return train_dataloader, test_dataloader, class_names
