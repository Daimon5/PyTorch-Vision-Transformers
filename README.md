# PyTorch-Vision-Transformers

This repository contains an implementation of a Vision Transformer in PyTorch for classifying images as Pizza, Steak, or Sushi. The project is organized into several scripts and a Jupyter notebook demonstrating the complete data download process to model training and evaluation.

## Contents

- `get_data.py`: Script containing the `download_data` function, which downloads and stores the required dataset directly from GitHub to a specified destination path.

- `data_setup.py`: Script containing the `create_dataloaders` function, which creates and returns the training and testing PyTorch dataloaders along with the class names.

- `engine.py`: Script containing the `train` function, which trains the given model using the dataloaders. Users are encouraged to customize this function based on their specific needs.

- `loss_curve_plotter.py`: Script containing the `plot_loss_curves` function, which plots the loss curves based on the results dictionary obtained from training.

- `Vision_Transformer_Implementation.ipynb`: A Jupyter notebook demonstrating how to use the above scripts to create and train a Vision Transformer from scratch.

## Requirements

To run the code in this repository, you will need the following dependencies:

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- Jupyter Notebook

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/Daimon5/PyTorch-Vision-Transformers.git
    ```
2. Navigate to the cloned directory:
    ```bash
    cd Vision_Transformer_Food_Classification
    ```
3. Download the dataset:
    ```bash
    python get_data.py --destination_path ./data/
    ```
4. Open the notebook:
    ```bash
    jupyter notebook Vision_Transformer_Implementation.ipynb
    ```
5. Run the notebook cells to load data, train the model, and visualize the results.

## How to Use

- **Data Download**:
    - Use the `download_data` function in `get_data.py` to download the dataset to your local machine.

- **Data Setup**:
    - Use the `create_dataloaders` function in `data_setup.py` to create train and test dataloaders.

- **Model Training**:
    - Train the Vision Transformer model using the `train` function in `engine.py`. Customize the training loop as needed for your specific use case.

- **Plotting Loss Curves**:
    - Use the `plot_loss_curves` function in `loss_curve_plotter.py` to visualize the loss curves after training.

## References

- [Vision Transformer (ViT) Paper](https://arxiv.org/abs/2010.11929)
- [PyTorch Documentation](https://pytorch.org/docs/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

