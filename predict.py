import argparse
import os
import cv2
from pathlib import Path
from ultralytics import YOLO
import utils

# Class configuration
CLASS_NAMES = {0: "turbine_blade", 1: "crack", 2: "burn"}
COLORS = {
    "turbine_blade": (255, 120, 0),  # Steel Blue
    "crack": (0, 0, 255),           # Neon Red
    "burn": (0, 165, 255)           # Glowing Amber / Orange
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO11 Inference on Borescope Images")
    parser.add_argument("image", type=str, help="Path to input borescope image")
    parser.add_argument("--model", type=str, default=None, help="Path to custom YOLO11 model weights (e.g. best.pt)")
    parser.add_argument("--output", type=str, default="output_annotated.jpg", help="Path to save annotated image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Verify input image exists
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: Input image '{args.image}' not found.")
        return
        
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Failed to read image from '{args.image}'")
        return
        
    height, width, _ = img.shape
    
    # Choose weights path
    model_path = args.model
    is_mock = False
    
    if model_path is None:
        # Search for default checkpoints
        default_trained = Path("runs/segment/train_turbine_defects/weights/best.pt")
        if default_trained.exists():
            model_path = str(default_trained)
        else:
            # Check if base weights exist
            if os.path.exists("yolo11n-seg.pt"):
                model_path = "yolo11n-seg.pt"
            else:
                is_mock = True
                
    if is_mock:
        print("="*60)
        print("WARNING: No trained model weights found (best.pt / yolo11n-seg.pt).")
        print("Running in DEMO/MOCK mode using synthetic defect overlays for testing.")
        print("To run actual inference, train your model or supply --model path/to/weights.pt.")
        print("="*60 + "\n")
        
        metrics, annotated_img = utils.run_mock_inference(img, COLORS)
    else:
        print(f"Loading YOLO11 model from {model_path}...")
        try:
            model = YOLO(model_path)
            print(f"Running inference on '{img_path.name}' (ImgSz: {width}x{height}, Conf: {args.conf})...")
            
            results = model(str(img_path), conf=args.conf)[0]
            
            # Analyze using spatial geometry utils
            metrics, blades, defects, unassociated = utils.analyze_detections(
                results.boxes, results.masks, CLASS_NAMES, COLORS, width, height
            )
            
            # Draw overlays
            annotated_img = utils.draw_overlays(img, blades, defects, unassociated, COLORS)
            metrics["is_mock"] = False
            
        except Exception as e:
            print(f"Failed to run model inference: {e}")
            print("Falling back to demo/mock inference...")
            metrics, annotated_img = utils.run_mock_inference(img, COLORS)
            
    # Save the output image
    out_path = Path(args.output)
    cv2.imwrite(str(out_path), annotated_img)
    print(f"\nSuccess: Annotated image saved to: {out_path.resolve()}")
    
    # Output metrics to console
    print("\n" + "="*40)
    print("        DETECTION REPORT")
    print("="*40)
    print(f"Source Image:       {img_path.name}")
    print(f"Total Blades:       {metrics['total_blades']}")
    print(f"Total Defects:      {metrics['total_defects']}")
    print(f" - Cracks:          {metrics['cracks_count']}")
    print(f" - Burns:           {metrics['burns_count']}")
    print("-" * 40)
    
    for blade in metrics["blades"]:
        print(f"Blade #{blade['blade_id']} (Conf: {blade['confidence']:.2f}) -> Status: {blade['status']}")
        if len(blade["defects"]) == 0:
            print("  No defects detected.")
        else:
            for d in blade["defects"]:
                print(f"  - {d['type'].upper()} ({d['severity']} Severity): compromised {d['percent_compromised']}% of blade area.")
                
    if "unassociated_defects" in metrics and metrics["unassociated_defects"]:
        print("-" * 40)
        print("Unassociated Defects (Outside Blade Boundaries):")
        for d in metrics["unassociated_defects"]:
            print(f"  - {d['type'].upper()} (Conf: {d['confidence']:.2f}, Area: {d['area_pixels']} px)")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
