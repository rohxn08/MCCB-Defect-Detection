
from paddleocr import PaddleOCR
import cv2
import re
import os
import numpy as np
import json

class RobustMCCBPipeline:
    def __init__(self):
        # Initialize PaddleOCR once
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    def detect_switch_y(self, img):
        """
        Detects the Y-coordinate of the top and bottom of the switch.
        Returns: (switch_top_y, switch_bottom_y) or (None, None)
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Inverted threshold to find dark switch
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        switch_contour = None
        max_area = 0
        
        for c in contours:
            area = cv2.contourArea(c)
            x, y, cw, ch = cv2.boundingRect(c)
            cx, cy = x + cw/2, y + ch/2
            
            # Switch constraints
            if (0.2 * w < cx < 0.8 * w) and (0.3 * h < cy < 0.8 * h) and (area > 5000):
                if area > max_area:
                    max_area = area
                    switch_contour = c
                    
        if switch_contour is not None:
            _, sy, _, sh = cv2.boundingRect(switch_contour)
            return sy, sy + sh
        return None, None

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

        # 2. Detect Switch (The Separator)
        switch_top, switch_bottom = self.detect_switch_y(img)
        
        h, w = img.shape[:2]
        
        # Define Regions
        if switch_top:
            table_y_max = switch_top
            
            # Rating can be ON the switch (XT-series) or BELOW it (P-series)
            rating_y_min = switch_top
            rating_y_max = int(h * 0.95)

            # Dynamically detect where the table starts by finding the
            # first known electrical-parameter header (Ue, Icu, Ics, Ui, Uimp).
            # Everything above that line is the top section (product name, S/N).
            table_header_pattern = re.compile(
                r'^(Ue|Icu|Ics|Ui|Uimp)\b', re.IGNORECASE
            )
            # Collect candidate lines above the switch, sorted by Y
            above_switch = []
            for line in all_lines:
                box = line[0]
                text = line[1][0]
                cy = sum([p[1] for p in box]) / 4
                if cy < switch_top:
                    above_switch.append({"text": text, "cy": cy})
            above_switch.sort(key=lambda k: k['cy'])

            # Find the Y of the first table-header keyword
            table_y_min = int(h * 0.05)  # fallback
            for item in above_switch:
                if table_header_pattern.match(item['text'].strip()):
                    table_y_min = max(0, int(item['cy']) - 20)  # small margin above
                    break
        else:
            # Fallback
            print("  Warning: Switch not detected. Using full image search.")
            table_y_min, table_y_max = 0, h
            rating_y_min, rating_y_max = 0, h

        # --- VISUALIZATION START ---
        debug_dir = os.path.join(os.path.dirname(image_path), "debug_pipeline")
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
            
        vis_img = img.copy()
        # Draw Switch (if found)
        if switch_top:
            cv2.line(vis_img, (0, switch_top), (w, switch_top), (0, 0, 255), 5)
            cv2.line(vis_img, (0, switch_bottom), (w, switch_bottom), (0, 0, 255), 5)
            cv2.putText(vis_img, "SWITCH AREA", (10, switch_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Draw Top Section ROI (Yellow)
        cv2.rectangle(vis_img, (5, 0), (w-5, table_y_min), (0, 255, 255), 4)
        cv2.putText(vis_img, "TOP ROI", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

        # Draw Table ROI (Green)
        cv2.rectangle(vis_img, (10, table_y_min), (w-10, table_y_max), (0, 255, 0), 4)
        cv2.putText(vis_img, "TABLE ROI", (50, table_y_min + 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # Draw Rating ROI (Blue)
        cv2.rectangle(vis_img, (20, rating_y_min), (w-20, rating_y_max), (255, 0, 0), 4)
        cv2.putText(vis_img, "RATING ROI", (50, rating_y_max - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        cv2.imwrite(os.path.join(debug_dir, f"debug_{filename}"), vis_img)
        # --- VISUALIZATION END ---

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
        print("ROBUST PIPELINE RESULTS")
        print("="*50)
        
        for f in files:
            data = pipeline.process_image(os.path.join(input_dir, f))
            if data:
                print(f"\nFile: {data['filename']}")
                print(f"Rating (ROI-filtered): {data['rating']}")
                print("Table Structure:")
                for r in data['table_data']:
                    print(f"  {r}")
    else:
        print("Input directory not found.")
