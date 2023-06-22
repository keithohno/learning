import cv2

image = cv2.imread("source.png")


# blur filters
image_gaussian_blur = cv2.GaussianBlur(image, (25, 25), 0)
cv2.imwrite("out/gaussian_blur.png", image_gaussian_blur)

image_median_blur = cv2.medianBlur(image, 25)
cv2.imwrite("out/median_blur.png", image_median_blur)

image_bilateral_blur = cv2.bilateralFilter(image, 9, 25, 25)
cv2.imwrite("out/bilateral_blur.png", image_bilateral_blur)
