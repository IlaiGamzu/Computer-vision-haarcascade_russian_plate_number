import cv2
import methods
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(r'https://github.com/IlaiGamzu/Computer-vision-haarcascade_russian_plate_number/blob/main/car_plate.jpg')
if image is None:
    print ("Your image not exist")
    exit(1)
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load the license plate cascade classifier
plate_cascade = cv2.CascadeClassifier(r'https://github.com/IlaiGamzu/Computer-vision-haarcascade_russian_plate_number/blob/main/haarcascade_russian_plate_number.xml')

# Display the image
methods.display(img_rgb)


# Detect and display the license plate
result = methods.detect_plate(img_rgb, plate_cascade)
methods.display(result)

# Detect and blur the license plate
result_blur = methods.detect_and_blur_plate(img_rgb, plate_cascade)
methods.display(result_blur)

# Show the results using plt.show() or any other desired display method
#plt.show()
