import cv2
import numpy as np

def main():
    # Similarity results and loop truth values
    with open("../data/scores.txt", "r") as fdata, open("./loopinf/loopinf-kitti-05.txt", "r") as floopinf:
        lines_data = fdata.readlines()
        lines_loopinf = floopinf.readlines()

    M = int(lines_data[0].split()[0])
    N = int(lines_data[0].split()[1])

    data = np.zeros((M, N))
    loopinf = []
    lcnum = 0

    for i in range(1, M + 1):
        row = list(map(float, lines_data[i].split()))
        data[i - 1] = row

    for line in lines_loopinf:
        currFrame, lcFrame = map(int, line.split('\t'))
        loopinf.append((currFrame, lcFrame))
        if lcFrame > -1:
            lcnum += 1

    thredCur = 0.0
    scale = 0.0001 #precision parameter
    precision = 0.0
    recall = 0.0
    maxPre = 0.0
    maxRec = 0.0

    while thredCur <= 1.0:
        lcdnum = 0
        truelcd = 0
        alllcd = 0

        for i in range(M):
            lcd_frame = []

            for j in range(N):
                if data[i][j] >= thredCur and abs(i - j) > 50:
                    lcd_frame.append(j)

            for lcd in lcd_frame:
                if loopinf[i][1] > -1:
                    alllcd += 1
                    if abs(loopinf[i][1] - lcd) < 50:
                        truelcd += 1

            if lcd_frame and loopinf[i][1] > -1:
                lcdnum += 1

        if lcnum != 0 and alllcd != 0 and (abs(recall - (float(lcdnum) / lcnum)) > 0.005 or abs(precision - (float(truelcd) / alllcd)) > 0.005):
            recall = float(lcdnum) / lcnum
            precision = float(truelcd) / alllcd

            if np.isnan(precision):
                break

            if recall == 1.0:
                maxPre = max(maxPre, precision)
            if precision == 1.0:
                maxRec = max(maxRec, recall)
            print('Precision: %.3f'%precision, 'Recall: %.3f'%recall)
        thredCur += scale

    print('Max Precision: %.3f'%maxPre, 'Max Recall: %.3f'%maxRec)

if __name__ == "__main__":
    main()

