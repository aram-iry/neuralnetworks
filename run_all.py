"""
One-command entry point: train → visualize → predict.
  python run_all.py
"""

from train import main as train_main
from visualize import plot_curves, plot_confusion_matrix
from predict import predict


if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 1 / 3 — TRAINING")
    print("=" * 60)
    train_main()

    print("\n" + "=" * 60)
    print("  STEP 2 / 3 — VISUALIZATION")
    print("=" * 60)
    plot_curves()
    plot_confusion_matrix()

    print("\n" + "=" * 60)
    print("  STEP 3 / 3 — INFERENCE & SUBMISSION")
    print("=" * 60)
    predict(use_tta=True)

    print("\n✅ Pipeline complete!")
