import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11 Instance Segmentation on Turbine Blade Defects")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml file")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt", help="Pretrained model weights (e.g., yolo11n-seg.pt, yolo11s-seg.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (use -1 for auto-batching)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g., cuda:0, cpu). Auto-detects GPU if not specified.")
    parser.add_argument("--workers", type=int, default=8, help="Number of data loader workers")
    parser.add_argument("--project", type=str, default="runs/segment", help="Project folder name")
    parser.add_argument("--name", type=str, default="train_turbine_defects", help="Experiment name")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Validate dataset configuration path
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"Error: Dataset configuration file not found at: {data_path}")
        print("Please check that the 'dataset/data.yaml' file exists and is correctly structured.")
        sys.exit(1)
        
    print(f"Using dataset config: {data_path}")
    
    # 2. Select Device
    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    print(f"Training device: {device} (CUDA available: {torch.cuda.is_available()})")
    if torch.cuda.is_available() and "cpu" not in device:
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")

    # 3. Load YOLO11 Segment model
    print(f"Loading base YOLO11 segmentation model: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Failed to load model {args.model}. Error: {e}")
        print("Falling back to downloading standard yolo11n-seg.pt...")
        model = YOLO("yolo11n-seg.pt")

    # 4. Start training
    print("Starting training pipeline...")
    try:
        results = model.train(
            data=str(data_path),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            exist_ok=True,
            plots=True  # Generates training validation plots automatically
        )
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Model saved to project: {args.project}/{args.name}")
        print(f"Best weights located at: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        print("\nTroubleshooting tips:")
        print("1. Verify your images and labels exist in the path designated by dataset/data.yaml.")
        print("2. Ensure labels contain valid class indices (0: turbine_blade, 1: crack, 2: burn).")
        print("3. For instance segmentation, label coordinates should be polygon outlines normalized to [0, 1].")
        sys.exit(1)

if __name__ == "__main__":
    main()
