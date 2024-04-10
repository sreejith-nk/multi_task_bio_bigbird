import torch

print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

def test_gpu():
    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        # Device configuration
        device = torch.device('cuda')
        print('GPU is available')
    else:
        device = torch.device('cpu')
        print('GPU is not available, running on CPU')

    # Create a random tensor and move it to the device
    x = torch.rand(5, 3).to(device)

    # Perform a simple operation on the tensor
    y = torch.ones_like(x)

    # Move the result back to the CPU if necessary
    y = y.to('cpu')

    # Print the result
    print('Result tensor:')
    print(y)

# Call the test function
test_gpu()
