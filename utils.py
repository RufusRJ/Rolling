import cv2
import numpy as np

def box_intersection(boxA, boxB):
    """
    Calculates what fraction of boxA lies inside boxB.
    Boxes are in format [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    if boxAArea == 0:
        return 0.0
        
    return interArea / boxAArea

def get_polygon_area(pts):
    """
    Calculates the pixel area of a polygon.
    """
    if pts is None or len(pts) < 3:
        return 0.0
    return cv2.contourArea(np.array(pts, dtype=np.float32))

def draw_overlays(img, blades, defects, unassociated_defects, colors):
    """
    Draws custom translucent segment overlays and bounding boxes onto the image.
    - Turbine Blades: Steel Blue
    - Cracks: Neon Red
    - Burns: Amber/Orange
    """
    overlay = img.copy()
    
    # 1. Draw Turbine Blades (Steel Blue)
    for b in blades:
        color = colors.get("turbine_blade", (255, 120, 0))
        pts = np.array(b["polygon"], dtype=np.int32)
        # Fill polygon on overlay
        cv2.fillPoly(overlay, [pts], color)
        # Outline contour on base image
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        # Draw bounding box
        x1, y1, x2, y2 = map(int, b["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
        # Text label
        label_text = f"Blade #{b['id']} ({b['confidence']:.2f})"
        cv2.putText(img, label_text, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, lineType=cv2.LINE_AA)

    # 2. Draw Defects (Cracks and Burns)
    all_defects = defects + unassociated_defects
    for d in all_defects:
        label = d["label"]
        color = colors.get(label, (0, 0, 255))
        
        pts = np.array(d["polygon"], dtype=np.int32)
        # Fill polygon on overlay
        cv2.fillPoly(overlay, [pts], color)
        # Outline contour on base image
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        
        # Display label near defect box
        x1, y1, _, _ = map(int, d["box"])
        pct_text = f" ({d['percent_compromised']:.1f}%)" if "percent_compromised" in d else ""
        label_text = f"{label.upper()} {d['confidence']:.2f}{pct_text}"
        cv2.putText(img, label_text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)

    # Blend overlay with original image (alpha = 0.35)
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

def analyze_detections(boxes, masks, class_names, colors, width, height):
    """
    Performs spatial analysis to group cracks/burns with corresponding turbine blades.
    Calculates defect areas, compromised percentages, and severity.
    """
    blades = []
    defects = []
    
    if boxes is not None:
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(box.cls[0].item())
            label = class_names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0].item())
            
            # Bounding box coords [x1, y1, x2, y2]
            xyxy = box.xyxy[0].tolist()
            
            # Polygon segment coords
            polygon = None
            if masks is not None and len(masks) > i:
                xy_pts = masks[i].xy
                if len(xy_pts) > 0 and len(xy_pts[0]) > 0:
                    polygon = xy_pts[0].tolist()
            
            if polygon is None:
                # Approximate polygon with bounding box corners
                polygon = [
                    [xyxy[0], xyxy[1]],
                    [xyxy[2], xyxy[1]],
                    [xyxy[2], xyxy[3]],
                    [xyxy[0], xyxy[3]]
                ]
            
            area = get_polygon_area(polygon)
            if area == 0:
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                
            det_info = {
                "id": len(blades) if label == "turbine_blade" else len(defects),
                "label": label,
                "confidence": conf,
                "box": xyxy,
                "polygon": polygon,
                "area_pixels": area
            }
            
            if label == "turbine_blade":
                det_info["defects"] = []
                blades.append(det_info)
            elif label in ["crack", "burn"]:
                defects.append(det_info)
                
    # Spatial grouping
    unassociated_defects = []
    for d in defects:
        best_blade = None
        max_overlap = 0.0
        
        for b in blades:
            overlap = box_intersection(d["box"], b["box"])
            if overlap > max_overlap:
                max_overlap = overlap
                best_blade = b
                
        if best_blade is not None and max_overlap > 0.10:
            d["blade_id"] = best_blade["id"]
            d["percent_compromised"] = (d["area_pixels"] / best_blade["area_pixels"]) * 100 if best_blade["area_pixels"] > 0 else 0
            best_blade["defects"].append(d)
        else:
            unassociated_defects.append(d)
            
    # Compile summary report dict
    summary = {
        "total_blades": len(blades),
        "total_defects": len(defects),
        "cracks_count": sum(1 for d in defects if d["label"] == "crack"),
        "burns_count": sum(1 for d in defects if d["label"] == "burn"),
        "blades": []
    }
    
    for b in blades:
        blade_defects = []
        for d in b["defects"]:
            # Classify severity based on compromised area ratio
            if d["percent_compromised"] < 0.5:
                severity = "Low"
            elif d["percent_compromised"] < 2.0:
                severity = "Medium"
            else:
                severity = "High"
                
            blade_defects.append({
                "defect_id": d["id"],
                "type": d["label"],
                "confidence": d["confidence"],
                "area_pixels": round(d["area_pixels"], 1),
                "percent_compromised": round(d["percent_compromised"], 2),
                "severity": severity
            })
            
        summary["blades"].append({
            "blade_id": b["id"],
            "confidence": b["confidence"],
            "box": [round(c, 1) for c in b["box"]],
            "area_pixels": round(b["area_pixels"], 1),
            "defects": blade_defects,
            "defect_count": len(blade_defects),
            "status": "Failed" if any(sd["severity"] == "High" for sd in blade_defects) else ("Action Required" if len(blade_defects) > 0 else "Passed")
        })
        
    if unassociated_defects:
        summary["unassociated_defects"] = []
        for d in unassociated_defects:
            summary["unassociated_defects"].append({
                "defect_id": d["id"],
                "type": d["label"],
                "confidence": d["confidence"],
                "area_pixels": round(d["area_pixels"], 1)
            })
            
    return summary, blades, defects, unassociated_defects

def run_mock_inference(img, colors):
    """
    Simulates predictions on an image to allow testing the UI without a trained model weights file.
    Detects 3 blades, a crack on blade 0, and a burn on blade 1.
    """
    h, w, _ = img.shape
    
    # 1. Define blades
    b1_poly = [[int(w * 0.15), int(h * 0.15)], [int(w * 0.35), int(h * 0.13)], [int(w * 0.32), int(h * 0.85)], [int(w * 0.12), int(h * 0.82)]]
    b2_poly = [[int(w * 0.40), int(h * 0.12)], [int(w * 0.60), int(h * 0.11)], [int(w * 0.57), int(h * 0.88)], [int(w * 0.37), int(h * 0.85)]]
    b3_poly = [[int(w * 0.65), int(h * 0.14)], [int(w * 0.85), int(h * 0.16)], [int(w * 0.82), int(h * 0.83)], [int(w * 0.62), int(h * 0.81)]]
    
    blades = [
        {"id": 0, "label": "turbine_blade", "confidence": 0.94, "box": [w * 0.12, h * 0.13, w * 0.35, h * 0.85], "polygon": b1_poly, "area_pixels": get_polygon_area(b1_poly), "defects": []},
        {"id": 1, "label": "turbine_blade", "confidence": 0.96, "box": [w * 0.37, h * 0.11, w * 0.60, h * 0.88], "polygon": b2_poly, "area_pixels": get_polygon_area(b2_poly), "defects": []},
        {"id": 2, "label": "turbine_blade", "confidence": 0.91, "box": [w * 0.62, h * 0.14, w * 0.85, h * 0.83], "polygon": b3_poly, "area_pixels": get_polygon_area(b3_poly), "defects": []}
    ]
    
    # 2. Define defects
    c1_poly = [[int(w * 0.20), int(h * 0.30)], [int(w * 0.22), int(h * 0.38)], [int(w * 0.21), int(h * 0.48)], 
               [int(w * 0.23), int(h * 0.48)], [int(w * 0.24), int(h * 0.38)], [int(w * 0.22), int(h * 0.30)]]
    crack = {
        "id": 0, "label": "crack", "confidence": 0.88, "box": [w * 0.20, h * 0.30, w * 0.24, h * 0.48], "polygon": c1_poly,
        "area_pixels": get_polygon_area(c1_poly), "blade_id": 0, "percent_compromised": 0.0
    }
    
    # Burn polygon (circle-like)
    burn_center = (int(w * 0.48), int(h * 0.50))
    burn_r = int(w * 0.04)
    burn_poly = []
    for angle in range(0, 360, 45):
        rad = np.deg2rad(angle)
        burn_poly.append([int(burn_center[0] + burn_r * np.cos(rad)), int(burn_center[1] + burn_r * np.sin(rad))])
    
    burn = {
        "id": 1, "label": "burn", "confidence": 0.82, "box": [burn_center[0] - burn_r, burn_center[1] - burn_r, burn_center[0] + burn_r, burn_center[1] + burn_r], "polygon": burn_poly,
        "area_pixels": get_polygon_area(burn_poly), "blade_id": 1, "percent_compromised": 0.0
    }
    
    # Associate
    crack["percent_compromised"] = (crack["area_pixels"] / blades[0]["area_pixels"]) * 100
    blades[0]["defects"].append(crack)
    
    burn["percent_compromised"] = (burn["area_pixels"] / blades[1]["area_pixels"]) * 100
    blades[1]["defects"].append(burn)
    
    defects = [crack, burn]
    unassociated = []
    
    # Create overlays
    annotated_img = draw_overlays(img, blades, defects, unassociated, colors)
    
    # Format summary JSON
    summary = {
        "total_blades": len(blades),
        "total_defects": len(defects),
        "cracks_count": 1,
        "burns_count": 1,
        "blades": [],
        "is_mock": True # Indicator that the server ran in demo mode
    }
    
    for b in blades:
        blade_defects = []
        for d in b["defects"]:
            if d["percent_compromised"] < 0.5:
                severity = "Low"
            elif d["percent_compromised"] < 2.0:
                severity = "Medium"
            else:
                severity = "High"
                
            blade_defects.append({
                "defect_id": d["id"],
                "type": d["label"],
                "confidence": d["confidence"],
                "area_pixels": round(d["area_pixels"], 1),
                "percent_compromised": round(d["percent_compromised"], 2),
                "severity": severity
            })
            
        summary["blades"].append({
            "blade_id": b["id"],
            "confidence": b["confidence"],
            "box": [round(c, 1) for c in b["box"]],
            "area_pixels": round(b["area_pixels"], 1),
            "defects": blade_defects,
            "defect_count": len(blade_defects),
            "status": "Failed" if any(sd["severity"] == "High" for sd in blade_defects) else ("Action Required" if len(blade_defects) > 0 else "Passed")
        })
        
    return summary, annotated_img
