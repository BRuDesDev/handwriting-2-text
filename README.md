

### Preprocess phone photo of a notebook page:
    - convert to grayscale
    - reduce shadows / uneven lighting
    - boost contrast
    - light denoise
    - optional thresholding


Phone photo of lined notebook page → flatten & clean → segment into text lines 
(or blocks) → OCR each line → stitch into paragraphs → export .txt + .json.
### Preprocess phone photo of a notebook page:
    - convert to grayscale
    - reduce shadows / uneven lighting
    - boost contrast
    - light denoise
    - optional thresholding

### Segment into text lines (or blocks):
    - use OpenCV to find contours or Hough lines
    - group contours into lines based on proximity and alignment
    - extract bounding boxes for each line/block    
### OCR each line:
    - use Tesseract OCR with appropriate configuration for handwriting
    - consider training a custom Tesseract model if handwriting is very unique
    - post-process OCR output to correct common errors (e.g., '0' vs 'O', '1' vs 'I')
### Stitch into paragraphs:
    - analyze line spacing and indentation to group lines into paragraphs
    - use simple heuristics (e.g., blank lines, indentation) to determine paragraph breaks
### Export .txt + .json:
    - save the extracted text in a .txt file
    - create a .json file with metadata (e.g., line bounding boxes, confidence scores, original image path)

### Example code snippet for preprocessing and OCR


```python
import cv2
import pytesseract
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding to reduce shadows
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    return processed
def ocr_image(image):
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)     
    return text         
# Example usage
image_path = 'notebook_page.jpg'
processed_image = preprocess_image(image_path)
extracted_text = ocr_image(processed_image)
print(extracted_text)
```                             