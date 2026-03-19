from train import main as train_main
from predict import predict

if __name__ == "__main__":
    print("--- Phase 1: Training ---")
    train_main()
    print("\n--- Phase 2: Inference ---")
    predict()