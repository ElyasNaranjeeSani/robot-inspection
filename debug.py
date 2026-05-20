import inspect
from anomalib.models.image.patchcore import torch_model
print(inspect.getsource(torch_model.PatchcoreModel.forward))