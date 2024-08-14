from ultralytics import YOLO
import cv2

model = YOLO('last (1).pt')

def detect_with_yolo(img):
    color_palette = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 165, 255), (128, 0, 128), (255, 255, 0), (255, 128, 0)]

    results = model.predict(img, stream=True)

    detected_img = img.copy()  # Copy of the original image
    table_roi = "not detected"
    company_roi = "not detected"
    number_roi = "not detected"
    date_roi = "not detected"

    for result in results:
        boxes = result.boxes.cpu().numpy()  # Get boxes on CPU in numpy format
        for box in boxes:  # Iterate over boxes
            r = box.xyxy[0].astype(int)  # Get corner points as int
            class_id = int(box.cls[0])  # Get class ID
            class_name = model.names[class_id]  # Get class name using the class ID
            print(f"Class: {class_name}, Box: {r}")  # Print class name and box coordinates
            # Draw boxes on the image
            color = color_palette[class_id]
            cv2.rectangle(detected_img, (r[0], r[1]), (r[2], r[3]), color, 2)
            
            # Check if the detected object is a specific class
            if class_name == 'table':
                table_roi = img[r[1]:r[3], r[0]:r[2]]  # Extract ROI for the table
            elif class_name == 'company':
                company_roi = img[r[1]:r[3], r[0]:r[2]]  # Extract ROI for the company
            elif class_name == 'number':
                number_roi = img[r[1]:r[3], r[0]:r[2]]  # Extract ROI for the number
            elif class_name == 'date':
                date_roi = img[r[1]:r[3], r[0]:r[2]]  # Extract ROI for the date
            print("TABLE ROI ", table_roi)
            print("COMPANY ROI ", company_roi)
            print("NUMBER ROI ", number_roi)
            print("DATE ROI ", date_roi)
    
    return detected_img, table_roi, company_roi, number_roi, date_roi