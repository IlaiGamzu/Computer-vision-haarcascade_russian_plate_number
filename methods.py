import cv2
import matplotlib.pyplot as plt

def display(img_rgb):
    fig= plt.figure (figsize=(10,10))
    ax_fig=fig.add_subplot(111)
    ax_fig.imshow(img_rgb)
    plt.show()
    if cv2.waitKey(0) & 0xFF== ord('q'):
        cv2.destroyAllWindows()

def detect_plate(img, plate_cascade):
    img_detect = img.copy()
    img_parameters = plate_cascade.detectMultiScale(img_detect,scaleFactor=1.3, minNeighbors=3)
    for (x, y, w, h) in img_parameters:
        cv2.rectangle(img_detect, (x, y), (x + w, y + h), (255, 0, 0))
    return img_detect

def detect_and_blur_plate(img, plate_cascade):
    img_detect = img.copy()
    img_parameters = plate_cascade.detectMultiScale(img_detect,scaleFactor=1.3, minNeighbors=3)
    for (x, y, w, h) in img_parameters:
        roi_img = img_detect[y:y + h, x:x + w]
        meidann_blur = cv2.medianBlur(roi_img, 7)
        img_detect[y:y + h, x:x + w] = meidann_blur
    return img_detect
