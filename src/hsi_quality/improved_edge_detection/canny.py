import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from collections import deque


class CannyDetector:
    def __init__(self, low_threshold_ratio: float = 0.1, high_threshold_ratio: float = 0.3, sigma: float = 1.0):
        self.low_threshold_ratio = low_threshold_ratio
        self.high_threshold_ratio = high_threshold_ratio
        self.sigma = sigma

    def detect_edge_pixels(self, img: np.ndarray, gsd_x: float, gsd_y: float) -> np.ndarray:
        # Step 1: Filter image to remove noise
        filtered_img = gaussian_filter(img, sigma=self.sigma)
        
        # Step 2: Find intensity gradient for each pixel
        gy = sobel(filtered_img, axis=0) / gsd_x
        gx = sobel(filtered_img, axis=1) / gsd_y

        mag = np.hypot(gx, gy)
        angle = np.arctan2(gy, gx)

        # Step 3: Non-maximum suppression
        nms = self.non_max_suppression(mag, angle, gx, gy)

        # Step 4: Hysteresis thresholding
        edge_pixels = self.hysteresis_thresholding(nms)

        return edge_pixels

    def non_max_suppression(self, mag: np.ndarray, angle: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        nms = np.zeros(mag.shape)
        
        for i in range(1, int(mag.shape[0]) - 1):
            for j in range(1, int(mag.shape[1]) - 1):
                if((angle[i,j] >= 0 and angle[i,j] <= 45) or (angle[i,j] < -135 and angle[i,j] >= -180)):
                    yBot = np.array([mag[i,j+1], mag[i+1,j+1]])
                    yTop = np.array([mag[i,j-1], mag[i-1,j-1]])
                    x_est = np.absolute(gy[i,j]/mag[i,j])
                    if (mag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and mag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                        nms[i,j] = mag[i,j]
                    else:
                        nms[i,j] = 0
                if((angle[i,j] > 45 and angle[i,j] <= 90) or (angle[i,j] < -90 and angle[i,j] >= -135)):
                    yBot = np.array([mag[i+1,j] ,mag[i+1,j+1]])
                    yTop = np.array([mag[i-1,j] ,mag[i-1,j-1]])
                    x_est = np.absolute(gx[i,j]/mag[i,j])
                    if (mag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and mag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                        nms[i,j] = mag[i,j]
                    else:
                        nms[i,j] = 0
                if((angle[i,j] > 135 and angle[i,j] <= 180) or (angle[i,j] < 0 and angle[i,j] >= -45)):
                    yBot = np.array([mag[i,j-1] ,mag[i+1,j-1]])
                    yTop = np.array([mag[i,j+1] ,mag[i-1,j+1]])
                    x_est = np.absolute(gy[i,j]/mag[i,j])
                    if (mag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and mag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                        nms[i,j] = mag[i,j]
                    else:
                        nms[i,j] = 0
        
        return nms
    
    def hysteresis_thresholding(self, img):
        h, w = img.shape
        high_threshold = self.high_threshold_ratio * np.max(img)
        low_threshold = self.low_threshold_ratio * high_threshold
        
        # 0 = non-edge
        # 1 = weak edge
        # 2 = strong edge
        sup = np.zeros((h, w), dtype=np.uint8)

        sup[(img >= low_threshold) & (img < high_threshold)] = 1
        sup[img >= high_threshold] = 2

        # Initialize queue with all strong edges
        queue = deque(zip(*np.where(sup == 2)))

        # 8-connected neighborhood
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)
        ]

        while queue:
            i, j = queue.popleft()

            for di, dj in neighbors:
                ni, nj = i + di, j + dj

                if 0 <= ni < h and 0 <= nj < w:
                    # Promote connected weak edges to strong
                    if sup[ni, nj] == 1:
                        sup[ni, nj] = 2
                        queue.append((ni, nj))

        # Keep only strong edges
        sup = (sup == 2).astype(np.uint8)

        return sup