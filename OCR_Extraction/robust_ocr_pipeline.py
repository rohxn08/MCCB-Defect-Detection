
from paddleocr import PaddleOCR
import cv2
import re
import os
import numpy as np
import json

from layout_detector import detect_layout_regions
from json_formatter import format_image_result

class RobustMCCBPipeline:
    def __init__(self):
        # Initialize PaddleOCR once
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    def extract_top_lines(self, ocr_results, y_max):
        """
        Collects raw OCR text lines whose vertical centre is above y_max.
        These are the top-section lines (logo, product name, serial number).
        Returns: list of strings sorted top-to-bottom.
        """
        top_items = []
        for line in ocr_results:
            box = line[0]
            text = line[1][0]
            cy = sum([p[1] for p in box]) / 4
            if cy < y_max:
                top_items.append({"text": text, "cy": cy})

        # Sort top-to-bottom and return just the text strings
        top_items.sort(key=lambda k: k['cy'])
        return [item['text'] for item in top_items]

    def extract_table(self, ocr_results, y_min, y_max):
        """
        Extracts and structures table data from OCR results within the vertical range [y_min, y_max].
        """
        table_boxes = []
        for line in ocr_results:
            box = line[0]
            text = line[1][0]
            cy = sum([p[1] for p in box]) / 4
            cx = sum([p[0] for p in box]) / 4
            h = abs(box[2][1] - box[0][1])
            
            if y_min < cy < y_max:
                table_boxes.append({"text": text, "cy": cy, "cx": cx, "h": h})
        
        if not table_boxes:
            return []

        # Virtual Grid Construction
        # 1. Sort by Y to find rows
        table_boxes.sort(key=lambda k: k['cy'])
        
        rows = []
        current_row = []
        last_y = table_boxes[0]['cy']
        avg_height = np.mean([b['h'] for b in table_boxes])
        y_tolerance = avg_height * 0.6
        
        for item in table_boxes:
            if abs(item['cy'] - last_y) <= y_tolerance:
                current_row.append(item)
            else:
                # Finish row
                current_row.sort(key=lambda k: k['cx'])
                rows.append([x['text'] for x in current_row])
                # Start new
                current_row = [item]
                last_y = item['cy']
        
        if current_row:
            current_row.sort(key=lambda k: k['cx'])
            rows.append([x['text'] for x in current_row])
            
        return rows

    def extract_rating_from_roi(self, ocr_results, y_min, y_max):
        """
        Scans specifically within the Rating ROI for 'In=...' pattern.
        """
        rating_pattern = re.compile(r'(?:I|l|1)n\s*=\s*(\d+\s*A)', re.IGNORECASE)
        
        for line in ocr_results:
            box = line[0]
            text = line[1][0]
            cy = sum([p[1] for p in box]) / 4
            
            if y_min < cy < y_max:
                match = rating_pattern.search(text)
                if match:
                    return match.group(0) # Return entire string like "In=63A"
        return "Not Found"

    def process_image(self, image_path):
        filename = os.path.basename(image_path)
        print(f"Processing {filename}...")
        
        img = cv2.imread(image_path)
        if img is None:
            return None

        # 1. OCR entire image once
        ocr_result = self.ocr.ocr(image_path, cls=True)
        if not ocr_result or ocr_result[0] is None:
            print("  No text found.")
            return None
        all_lines = ocr_result[0]

        # 2. Detect Layout Regions (via layout_detector)
        debug_dir = os.path.join(os.path.dirname(image_path), "debug_pipeline")
        regions = detect_layout_regions(image_path, debug_dir, ocr_lines=all_lines)
        
        h, w = img.shape[:2]
        
        if regions:
            # Use layout detector's ROI boundaries directly
            table_y_min, table_y_max = regions["table"]
            rating_y_min, rating_y_max = regions["rating"]
        else:
            # Fallback
            h = img.shape[0]
            print("  Warning: Switch not detected. Using full image search.")
            table_y_min, table_y_max = 0, h
            rating_y_min, rating_y_max = 0, h

        # 3. Extract Data
        # Top-section lines (product name, serial number, etc.)
        top_lines = self.extract_top_lines(all_lines, table_y_min)

        # Rating
        rating_val = self.extract_rating_from_roi(all_lines, rating_y_min, rating_y_max)
        
        # Table
        table_rows = self.extract_table(all_lines, table_y_min, table_y_max)
        
        return {
            "filename": filename,
            "top_lines": top_lines,
            "rating": rating_val,
            "table_data": table_rows
        }

if __name__ == "__main__":
    pipeline = RobustMCCBPipeline()
    
    # Adjust path
    project_root = os.getcwd()
    if "OCR_Extraction" in project_root:
        project_root = os.path.dirname(project_root)
        
    input_dir = os.path.join(project_root, "images", "master_mccb")
    
    if os.path.exists(input_dir):
        files = [f for f in os.listdir(input_dir) if f.startswith("master") and f.lower().endswith(('.png', '.jpg'))]
        
        print("\n" + "="*50)
        print("  ROBUST MCCB OCR PIPELINE")
        print("="*50)
        
        results = {}
        
        for f in files:
            image_path = os.path.join(input_dir, f)
            raw_data = pipeline.process_image(image_path)
            
            if raw_data is None:
                print(f"  ⚠ Skipped {f} (no data)")
                continue
            
            results[f] = format_image_result(raw_data)
        
        # Pretty print to console
        print("\n" + json.dumps(results, indent=2))
        
        # Save to file
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results.json"
        )
        with open(output_path, "w") as fh:
            json.dump(results, fh, indent=2)
        
        print(f"\n✓ Saved structured JSON to {output_path}")
    else:
        print("Input directory not found.")
