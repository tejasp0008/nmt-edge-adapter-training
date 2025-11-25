# src/__main__.py
import argparse
from .train import run_quick_demo_training
from ..scripts.package_workspace import package_workspace

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run demo training")
    parser.add_argument("--package", action="store_true", help="Package workspace zip")
    args = parser.parse_args()
    if args.demo:
        run_quick_demo_training()
    if args.package:
        package_workspace()

if __name__ == "__main__":
    main()
