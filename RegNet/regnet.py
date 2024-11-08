import torch
import io
from torchvision import models
from torch_mlir import fx

class CustomRegNet(torch.nn.Module):
    def __init__(self):
        super(CustomRegNet, self).__init__()
        self.regnet = models.regnet_y_400mf(pretrained=True)

    def forward(self, x):
        return self.regnet(x)

def save_model(model, output):
    bytecode_stream = io.BytesIO()
    model.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()
    with open(output, "wb") as file:
        file.write(bytecode)

x = torch.rand(1, 3, 224, 224)

'''regnet = models.regnet_y_400mf(pretrained=True)
regnet.eval()
predictions = regnet(x)'''

model = CustomRegNet()
model.eval()
#scripted_model = torch.jit.script(model)

m = fx.export_and_import(model, x, enable_ir_printing=True, enable_graph_printing=True)
save_model(m, "regnet.mlir")

mlir_text = "regnet_txt.mlir"
with open(mlir_text, "w") as f:
    f.write(str(m))
print(m)
