import numpy as np
from mss import mss
#from mss.windows import MSS as mss
from skimage.measure import block_reduce
from PIL import Image

class DinoImageUtil():
    def __init__(self):
        self.sct = mss()

    def get_monitor_dimensions(self, monitorId=1):
        return self.sct.monitors[monitorId]

    def get__processed_screenshot_np(self, area):
        img_sct = self.sct.grab(area)
        #Converting to numpy array returns RGBA value for each pixel. This: [:,:,0:1] just grabs the red channel
        img_np = np.array(img_sct)[:,:,0:1]
        #Convert to black and white
        img_np[img_np < 128] = 0
        img_np[img_np >= 128] = 1

        img_np = block_reduce(img_np, block_size=(4, 4, 1), func=np.mean)

        # #show image
        # img = Image.fromarray(np.uint8(img_np[:,:,0] * 255) , 'L')
        # img.show()

        return img_np


    def save_screenshot(self, screenshot, filename):
        img = Image.fromarray(np.uint8(screenshot[:,:,0] * 255) , 'L')
        img.save(filename)