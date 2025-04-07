import torch
from model import RegressionController

def load_model(checkpoint_path, input_size, hidden_size, output_size):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    model = RegressionController(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model.load_state_dict(state_dict) 
    model.eval()
    return model


def run_inference(model):
    print("Enter two floating-point numbers separated by a space (or type 'exit' to quit):")
    while True:
        user_input = input("Input: ")
        if user_input.lower() == 'exit':
            print("Exiting inference loop.")
            break
        try:
            inputs = list(map(float, user_input.split()))
            if len(inputs) != 2:
                print("Please enter exactly two numbers.")
                continue
            input_tensor = torch.tensor([inputs], dtype=torch.float32)
            output = model.model(input_tensor)
            print(f"Model Output: {output.detach().numpy()}")
        except ValueError:
            print("Invalid input. Please enter two floating-point numbers.")

if __name__ == "__main__":
    INPUT_SIZE = 2
    HIDDEN_SIZE = 3
    OUTPUT_SIZE = 12
    CHECKPOINT_PATH = "/Users/danh/Documents/GitHub/mlp-regression-controller/lightning_logs/version_13/checkpoints/epoch=333-step=1000.ckpt"  # Path to the saved model checkpoint

    model = load_model(CHECKPOINT_PATH, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    run_inference(model)