import cv2
import numpy as np

def main():
    fin = open("../data/scores.txt", "r")
    M, N = map(int, fin.readline().split())
    data = np.zeros((M, N))
    for i in range(M):
        for j, val in enumerate(map(float, fin.readline().split())):
            data[i, j] = val
    
    B = (data * 255).astype(np.uint8)
    B = cv2.applyColorMap(B, cv2.COLORMAP_JET)
    
    cv2.imshow("scores_img", B)
    cv2.imwrite("scores_img.bmp", B)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

