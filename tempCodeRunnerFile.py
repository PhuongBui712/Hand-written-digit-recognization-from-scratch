    image = cv2.resize(img_arr, (28, 28))/255
    image = np.pad(image, (10, 10), 'constant', constant_values=0)
    image = cv2.resize(image, (28, 28))/255