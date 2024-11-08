from transformers import RegNetConfig, RegNetModel
import torch
import io
#from torch_mlir import fx

def save_model(model, output):
    bytecode_stream = io.BytesIO()
    model.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()
    with open(output, "wb") as file:
        file.write(bytecode)

configuration = RegNetConfig()

model = RegNetModel(configuration)
x = torch.rand(1, 3, 224, 224)

#m = fx.export_and_import(model, x, enable_ir_printing=True, enable_graph_printing=True)
#save_model(m, "regnet.mlir")

m = torch.export.export(model, (x,))
export_text = "regnet_export.txt"
with open(export_text, "w") as f:
    f.write(str(m))
#print(m)
