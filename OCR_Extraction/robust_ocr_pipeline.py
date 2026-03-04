
from paddleocr import PaddleOCR
import cv2
import re
import os
import sys
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

        # 1. Rotate the image the same way layout_detector does 
        # so OCR coordinates align perfectly with ROI coordinates
        img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 2. OCR the rotated image array instead of the unrotated image path
        ocr_result = self.ocr.ocr(img_rotated, cls=True)
        if not ocr_result or ocr_result[0] is None:
            print("  No text found.")
            return None
        all_lines = ocr_result[0]

        # 2. Detect Layout Regions (via layout_detector)
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization")

        regions = detect_layout_regions(image_path, debug_dir, ocr_lines=all_lines)
        
        # Use shape of the rotated image because OCR ran on the rotated image!
        h, w = img_rotated.shape[:2]
        
        if regions:
            # Use layout detector's ROI boundaries directly
            table_y_min, table_y_max = regions["table"]
            rating_y_min, rating_y_max = regions["rating"]
        else:
            # Fallback
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
    
    print("\n" + "="*50)
    print("  ROBUST MCCB OCR PIPELINE")
    print("="*50)

    while True:
        try:
            choice = input("Do you want to process a single image (1) or a directory (2)? [1/2]: ").strip()
            if choice in ['1', '2']:
                break
            print("Invalid choice. Please enter 1 or 2.")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)

    # Adjust path
    project_root = os.getcwd()
    if "OCR_Extraction" in project_root:
        project_root = os.path.dirname(project_root)
        
    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    images_to_process = []
    
    if choice == '1':
        img_path = input("Enter the full path to the image: ").strip()
        # Remove surrounding quotes if dragged-and-dropped in terminal
        img_path = img_path.strip('\'"')
        if os.path.isfile(img_path):
            images_to_process.append(img_path)
        else:
            print(f"Invalid image path: {img_path}")
    elif choice == '2':
        default_dir = os.path.join(project_root, "images", "master_mccb")
        input_dir = input(f"Enter directory path [Press Enter for default: {default_dir}]: ").strip()
        if not input_dir:
            input_dir = default_dir
        else:
            input_dir = input_dir.strip('\'"')
            
        if os.path.isdir(input_dir):
            files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            images_to_process.extend([os.path.join(input_dir, f) for f in files])
        else:
            print(f"Directory not found: {input_dir}")
    else:
        print("Invalid choice. Please enter 1 or 2.")
        
    if images_to_process:
        print(f"\nFound {len(images_to_process)} image(s) to process.")
        
        for img_path in images_to_process:
            filename = os.path.basename(img_path)
            raw_data = pipeline.process_image(img_path)
            
            if raw_data is None:
                print(f"  ⚠ Skipped {filename} (no data extracted)")
                continue
            
            # Format JSON
            formatted_data = format_image_result(raw_data)
            
            # Determine filename
            serial_number = formatted_data.get("serial_number")
            
            # Clean serial number for filesystem if it exists
            if serial_number:
                # Remove any characters that might be invalid in filenames
                safe_sn = "".join(c for c in serial_number if c.isalnum() or c in ('-', '_')).strip()
                if safe_sn:
                    out_name = f"{safe_sn}.json"
                else:
                    out_name = f"{os.path.splitext(filename)[0]}.json"
            else:
                out_name = f"{os.path.splitext(filename)[0]}.json"
                
            out_path = os.path.join(output_dir, out_name)
            
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(formatted_data, fh, indent=2, ensure_ascii=False)
                
            print(f"\n  ✓ Result JSON:")
            print(json.dumps(formatted_data, indent=2, ensure_ascii=False))
            print(f"  ✓ Saved {filename} results to: {out_path}\n")
