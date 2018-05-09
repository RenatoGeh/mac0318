import cv2

class Camera(object):

    def __init__(self, 
                 height=120, 
                 width=160, 
                 input_cam_device=0):
        WIDTH_ID = 3
        HEIGHT_ID = 4
        self.height = height
        self.width = width
        self.input_cam_device = input_cam_device
        self.cam = cv2.VideoCapture(input_cam_device)
        print("Connecting to recording input device... ", end="", flush=True)
        while not self.cam.isOpened():
            pass
        print("Successful")
        self.cam.set(WIDTH_ID, width)
        self.cam.set(HEIGHT_ID, height)


    def __str__(self):
        cam_info = "height = {}\n".format(self.height)
        cam_info += "width = {}\n".format(self.width)
        cam_info += "capturing from device {}".format(self.input_cam_device)
        return cam_info

    def take_picture(self):
        _, img = self.cam.read()
        return img

    def save_picture(self, path, img):
        cv2.imwrite(path, img)


