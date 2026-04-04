import os
import sys
import subprocess
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run_step(script_name: str, *args: str) -> None:
    root = project_root()
    src_dir = root / "src"
    script_path = src_dir / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = [sys.executable, str(script_path), *args]

    print(f"\n▶ Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(src_dir), text=True, capture_output=True)

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print("STDERR:\n" + result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")


def main() -> None:
    root = project_root()
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)

    print("Submitting local pipeline on Local...")

    # Step 1: Feature engineering
    run_step("xgboost_feature_engineering.py")

    # Step 2: Training
    run_step("xgboost_train.py")

    # Step 3: Evaluation on the saved held-out test split
    run_step("xgboost_evaluate.py")

    # Step 4: Prediction on numbered CSVs only
    #input_dir = root / "inara_data"
    #run_step("predict.py", str(input_dir))

    print("\nPipeline completed successfully 🚀")


if __name__ == "__main__":
    main()