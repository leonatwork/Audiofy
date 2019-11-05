import cv2
import sys
import pytesseract
 
def resize(img, height=800):
  """ Resize image to given height """
  rat = height / img.shape[0]	    
  return cv2.resize(img, (int(rat * img.shape[1]), height))

if __name__ == '__main__':
 
  if len(sys.argv) < 2:
    print('Usage: python ocrImage.py image.jpg')
    sys.exit(1)
   
  # Read image path from command line
  imPath = sys.argv[1]
     
  # Uncomment the line below to provide path to tesseract manually
  # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
 
  # Define config parameters.
  # '-l eng'  for using the English language
  # '--oem 1' for using LSTM OCR Engine
  config = ('-l eng --oem 1 --psm 3')
 
  # Read image from disk
  im = cv2.imread(imPath, cv2.IMREAD_COLOR)

  # Load image and convert it from BGR to RGB
  image = cv2.cvtColor(cv2.imread(imPath), cv2.COLOR_BGR2RGB)
  
  # Resize and convert to grayscale
  img = cv2.cvtColor(resize(image), cv2.COLOR_BGR2GRAY)
  #cv2.imwrite("g1.jpg", img)

  # Run tesseract OCR on image
  text = pytesseract.image_to_string(image, config=config)
 
  # Write recognized text
  f = open("extractedTexts/recognizedText.txt", "w")
  f.write(text)
  f.close()
