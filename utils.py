
import torch
def save_model(model: torch.nn.Module, fname= str):
    """
    Saves a given PyTorch model

    Args:
        model: PyTorch model which is to be saved
    """

    torch.save(model, fname + '.pt')

def load_model(file_path):
    '''
    Loads a previously saved model

    Args:
        file_path: A string containing the path where the model is stored
    '''

    return torch.load(file_path)
