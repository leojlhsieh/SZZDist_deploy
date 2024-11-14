# %%
import torch
import sys
import logging

def check_gpu() -> None:
    logging.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logging.info("CUDA is available. Here are the details of the CUDA devices:")
        for i in range(torch.cuda.device_count()):
            logging.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
            a = f"  CUDA Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor},"
            b = f"  Multiprocessors: {torch.cuda.get_device_properties(i).multi_processor_count}"
            logging.info(f"{a+b}")
            logging.info(f"  Memory")
            logging.info(f"    {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB: Total Memory")
            logging.info(f"    {torch.cuda.memory_reserved(i) / (1024 ** 3):.2f} GB: PyTorch current Reserved Memory")
            logging.info(f"    {torch.cuda.memory_allocated(i) / (1024 ** 3):.2f} GB: PyTorch current Allocated Memory")
            logging.info(f"    {torch.cuda.max_memory_reserved(i) / (1024 ** 3):.2f} GB: PyTorch max ever Reserved Memory")
            logging.info(f"    {torch.cuda.max_memory_allocated(i) / (1024 ** 3):.2f} GB: PyTorch max ever Allocated Memory")

    else:
        logging.info("CUDA is NOT available")

if __name__ == "__main__":
    # print python version
    logging.basicConfig(level=logging.DEBUG)
    # logging.info = print
    logging.info(f"Python version: {sys.version}")
    check_gpu()
# %%
