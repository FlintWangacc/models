# TinyVGG.py
import torch
from torch import nn
#from torch_mlir import torchscript
import io
from torch_mlir import fx
# from torchvision.datasets import FashionMNIST
# from torchvision.transforms import ToTensor


class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

def save_model(model, output):
    bytecode_stream = io.BytesIO()
    model.operation.write_bytecode(bytecode_stream)
    bytecode =  bytecode_stream.getvalue()
    with open(output, "wb") as file:
        file.write(bytecode)

'''class tinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_state, ouput_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_state,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_state, out_channels=hidden_state,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_state, out_channels=hidden_state,
                      kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_state, out_channels=hidden_state,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_state*7*7, out_features=ouput_shape)
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        return (self.classifier_layer(x))'''

# train_data = FashionMNIST(root='./FashionMNIST',train=True, download=True, transform=ToTensor(), target_transform=None)
# class_names = train_data.classes


model = TinyVGG(input_shape=1, hidden_units=10, output_shape=10).to('cpu')
model.load_state_dict(torch.load(
    "/home/hmsjwzb/work/tinyVGG/tinyVGG.pth", weights_only=True))
parameters = model.state_dict()
for name, param in parameters.items():
    print(f"Layer name: {name}, Shape: {param.shape}")
onnx_file_path = "model.onnx"
torch.onnx.export(model, torch.randn(1, 1, 28, 28), onnx_file_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'])
# mlir_model = torchscript.compile(model, torch.ones(1, 1, 28, 28))
# save_model(mlir_model, "tinyVGG.mlir")
m = fx.export_and_import(model, torch.randn(1, 1, 28, 28), enable_ir_printing=True,
                        output_type = "linalg-on-tensors", enable_graph_printing=True)
print(m)
