import argparse
import os
import cv2
from pathlib import Path
from ultralytics import YOLO
import utils

# Class configuration
CLASS_NAMES = {0: "burn", 1: "crack", 2: "turbine_blade"}
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
    parser.add_argument("--calib-method", type=str, choices=["known_blade", "fixed_scale"], default="known_blade", help="Calibration method: known_blade or fixed_scale")
    parser.add_argument("--blade-height", type=float, default=100.0, help="Physical height of turbine blade in millimeters (used for known_blade method)")
    parser.add_argument("--fixed-scale", type=float, default=0.25, help="Fixed spatial resolution scale in mm/pixel (used for fixed_scale method)")
    parser.add_argument("--unit", type=str, choices=["mm", "cm", "m"], default="mm", help="Target display units")
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
        
        metrics, annotated_img = utils.run_mock_inference(
            img, COLORS,
            calibration_method=args.calib_method,
            known_blade_height=args.blade_height,
            fixed_scale=args.fixed_scale,
            target_unit=args.unit
        )
    else:
        print(f"Loading YOLO11 model from {model_path}...")
        try:
            model = YOLO(model_path)
            print(f"Running inference on '{img_path.name}' (ImgSz: {width}x{height}, Conf: {args.conf})...")
            
            results = model(str(img_path), conf=args.conf)[0]
            
            # Analyze using spatial geometry utils
            metrics, blades, defects, unassociated = utils.analyze_detections(
                results.boxes, results.masks, CLASS_NAMES, COLORS, width, height,
                calibration_method=args.calib_method,
                known_blade_height=args.blade_height,
                fixed_scale=args.fixed_scale,
                target_unit=args.unit
            )
            
            # Draw overlays
            annotated_img = utils.draw_overlays(img, blades, defects, unassociated, COLORS)
            metrics["is_mock"] = False
            
        except Exception as e:
            print(f"Failed to run model inference: {e}")
            print("Falling back to demo/mock inference...")
            metrics, annotated_img = utils.run_mock_inference(
                img, COLORS,
                calibration_method=args.calib_method,
                known_blade_height=args.blade_height,
                fixed_scale=args.fixed_scale,
                target_unit=args.unit
            )
            
    # Save the output image
    out_path = Path(args.output)
    cv2.imwrite(str(out_path), annotated_img)
    print(f"\nSuccess: Annotated image saved to: {out_path.resolve()}")
    
    # Output metrics to console
    print("\n" + "="*45)
    print("            DETECTION REPORT")
    print("="*45)
    print(f"Source Image:       {img_path.name}")
    print(f"Total Blades:       {metrics['total_blades']}")
    print(f"Total Defects:      {metrics['total_defects']}")
    print(f" - Cracks:          {metrics['cracks_count']}")
    print(f" - Burns:           {metrics['burns_count']}")
    print(f"Calibration Method: {args.calib_method}")
    print(f"Units Selected:     {args.unit} / {metrics['area_unit']}")
    print("-" * 45)
    
    for blade in metrics["blades"]:
        print(f"Blade #{blade['blade_id']} (Conf: {blade['confidence']:.2f}) -> Status: {blade['status']}")
        print(f"  Physical Height:  {blade['height_physical']} {metrics['length_unit']}")
        print(f"  Physical Area:    {blade['area_physical']} {metrics['area_unit']} ({blade['area_pixels']} px)")
        if len(blade["defects"]) == 0:
            print("  No defects detected.")
        else:
            for d in blade["defects"]:
                print(f"  - {d['type'].upper()} ({d['severity']} Severity): compromised {d['percent_compromised']}% of blade area.")
                print(f"    Physical Area:  {d['area_physical']} {metrics['area_unit']} ({d['area_pixels']} px)")
                print(f"    Max Dimension:  {d['max_dim_physical']} {metrics['length_unit']}")
                
    if "unassociated_defects" in metrics and metrics["unassociated_defects"]:
        print("-" * 45)
        print("Unassociated Defects (Outside Blade Boundaries):")
        for d in metrics["unassociated_defects"]:
            print(f"  - {d['type'].upper()} (Conf: {d['confidence']:.2f})")
            print(f"    Physical Area:  {d['area_physical']} {metrics['area_unit']} ({d['area_pixels']} px)")
            print(f"    Max Dimension:  {d['max_dim_physical']} {metrics['length_unit']}")
    print("="*45 + "\n")

if __name__ == "__main__":
    main()
