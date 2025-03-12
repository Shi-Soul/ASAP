import torch
import time
import argparse
from typing import Tuple

def exponential_backoff_matrix_multiplication(initial_size: Tuple[int, int], device: torch.device):
    """
    Perform matrix multiplication on the GPU with exponential backoff.
    
    Args:
        initial_size (Tuple[int, int]): Initial size of the matrices (rows, cols).
        device (torch.device): The GPU device to use.
    """
    size = initial_size
    while True:
        try:
            # Allocate matrices on the GPU
            print(f"Attempting to allocate matrices of size {size}...")
            A = torch.randn(size, device=device)
            B = torch.randn(size, device=device)
            
            # Perform matrix multiplication
            C = torch.matmul(A, B)
            torch.cuda.synchronize()  # Ensure the computation is done
            time.sleep(0.01)  # Small delay to avoid overwhelming the GPU
            size = (int(size[0] * 1.5), int(size[1] * 1.5))
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Reduce matrix size exponentially
                size = (size[0] // 2, size[1] // 2)
                print(f"Out of memory. Reducing matrix size to {size}...")
                if size[0] < 1 or size[1] < 1:
                    raise RuntimeError("Matrix size too small. GPU memory is insufficient.")
                torch.cuda.empty_cache()  # Clear GPU memory
            else:
                raise e

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Maximize GPU utilization using matrix multiplications.")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU to use (default: 0).")
    args = parser.parse_args()

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return
    
    # Set the GPU device
    device = torch.device(f"cuda:{args.gpu_id}")
    print(f"Using GPU: {torch.cuda.get_device_name(device)} (ID: {args.gpu_id})")
    
    # Initial matrix size (adjust based on your GPU memory)
    initial_size = (10000, 10000)  # Start with a large matrix
    
    try:
        exponential_backoff_matrix_multiplication(initial_size, device)
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()