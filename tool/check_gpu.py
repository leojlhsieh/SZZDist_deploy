# %%
import torch
import sys

def check_gpu() -> None:
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print("CUDA is available. Here are the details of the CUDA devices:")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  CUDA Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}", end=",")
            print(f"  Multiprocessors: {torch.cuda.get_device_properties(i).multi_processor_count}")
            print(f"  Memory")
            print(f"    {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB: Total Memory")
            print(f"    {torch.cuda.memory_reserved(i) / (1024 ** 3):.2f} GB: PyTorch current Reserved Memory")
            print(f"    {torch.cuda.memory_allocated(i) / (1024 ** 3):.2f} GB: PyTorch current Allocated Memory")
            print(f"    {torch.cuda.max_memory_reserved(i) / (1024 ** 3):.2f} GB: PyTorch max ever Reserved Memory")
            print(f"    {torch.cuda.max_memory_allocated(i) / (1024 ** 3):.2f} GB: PyTorch max ever Allocated Memory")

    else:
        print("CUDA is NOT available")

if __name__ == "__main__":
    # print python version
    print(f"Python version: {sys.version}")
    check_gpu()