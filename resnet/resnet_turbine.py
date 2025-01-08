import torch
import torchvision.models as models
from torch_mlir import fx
import iree.turbine.aot as aot
import io
import os

def save_model(model, output):
    bytecode_stream = io.BytesIO()
    model.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()
    with open(output, "wb") as file:
        file.write(bytecode)

# Step 1: Load a pretrained ResNet model
model = models.resnet50(pretrained=True)  # You can choose other ResNet variants as neededmodel.eval()

# Step 2: Create a dummy input tensor
dummy_input = torch.rand(1, 3, 224, 224)  # ResNet expects input size of (1, 3, 224, 224)

# Step 3: Trace the model
traced_model = torch.symbol.trace(model, dummy_input)

# Step 4: Convert the traced model to MLIR
m = fx.export_and_import(model, torch.randn(1, 3, 224, 224), enable_ir_printing=True,
                         enable_graph_printing=True)

# Step 5: Save the MLIR model to a file
mlir_file = "resnet50.mlir"


save_model(m, mlir_file)

print(f"MLIR model saved as '{mlir_file}'")

# Step 6: Use IREE to translate MLIR to VMFB
# Ensure that IREE is in your PATH
#os.system(f"iree-translate --mlir-to-vm-bytecode {mlir_file} -o resnet50.vmfb")

print("VMFB file generated as 'resnet50.vmfb'")

