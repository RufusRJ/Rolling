import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO11 Instance Segmentation Model")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml file")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights file. Defaults to best trained weights.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device to run evaluation on (cpu or cuda:0)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate on")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Resolve model path
    model_path = args.model
    if model_path is None:
        default_trained = Path("runs/segment/train_turbine_defects/weights/best.pt")
        if default_trained.exists():
            model_path = str(default_trained)
        else:
            print("Error: Trained model weights not found at runs/segment/train_turbine_defects/weights/best.pt")
            print("Please run 'python train.py' first to train the model, or specify a custom weight path using --model.")
            sys.exit(1)
            
    print(f"Loading model for evaluation: {model_path}")
    
    # 2. Check dataset configuration
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"Error: Dataset configuration file not found at: {data_path}")
        sys.exit(1)
        
    # 3. Setup device
    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    print(f"Running evaluation on device: {device}")
    
    # 4. Load YOLO model and run evaluation
    try:
        model = YOLO(model_path)
        print(f"Evaluating split '{args.split}' on model...")
        
        metrics = model.val(
            data=str(data_path),
            imgsz=args.imgsz,
            device=device,
            split=args.split,
            plots=True
        )
        
        # 5. Extract and print metrics
        print("\n" + "="*50)
        print("          EVALUATION RESULTS SUMMARY")
        print("="*50)
        
        # Box metrics
        print("Bounding Box Metrics:")
        print(f"  - Precision:        {metrics.results_dict['metrics/precision(B)']:.4f}")
        print(f"  - Recall:           {metrics.results_dict['metrics/recall(B)']:.4f}")
        print(f"  - mAP50:            {metrics.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"  - mAP50-95:         {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")
        print("-" * 50)
        
        # Segmentation Mask metrics
        print("Segmentation Mask Metrics:")
        print(f"  - Precision:        {metrics.results_dict['metrics/precision(M)']:.4f}")
        print(f"  - Recall:           {metrics.results_dict['metrics/recall(M)']:.4f}")
        print(f"  - mAP50:            {metrics.results_dict['metrics/mAP50(M)']:.4f}")
        print(f"  - mAP50-95:         {metrics.results_dict['metrics/mAP50-95(M)']:.4f}")
        print("="*50 + "\n")
        
        print("Detailed evaluation charts and confusion matrices have been saved in the validation directory.")
        
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        print("Ensure that the dataset paths defined in data.yaml are correct and images/labels exist in the split folders.")
        sys.exit(1)

if __name__ == "__main__":
    main()
