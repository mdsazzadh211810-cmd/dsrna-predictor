import torch
import torch_geometric
import numpy
import Bio

def test_environment():
    assert torch.__version__ == "2.5.1"
    assert torch.cuda.is_available() == True
    print(" all packages are all right!")

test_environment()