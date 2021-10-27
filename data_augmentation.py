import cv2

def to_gray(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return result

def to_blurred(image, kernel_sz=5):
    result = cv2.GaussianBlur(image, kernel_sz)
    return result

def to_outlines(image, th1=60, th2=180):
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    outlines = cv2.bitwise_not(cv2.Canny(image_bw, th1, th2))
    return outlines

def to_sketch(image):
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    invert_img = cv2.bitwise_not(image_bw)
    blur_img = cv2.GaussianBlur(invert_img, (21,21), 0)
    invblur_img = cv2.bitwise_not(blur_img)
    sketch_img = cv2.divide(image_bw, invblur_img, scale=256.0)
    return sketch_img

if __name__ == '__main__':
    test_img = cv2.imread('test.jpg')
    cv2.imshow("frame", to_sketch(test_img))
    cv2.waitKey(5000)