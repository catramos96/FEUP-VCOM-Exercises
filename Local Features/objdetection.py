import cv2 as cv
import numpy as np
import sys

import time

'''
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html
'''

def openImage(filename):
    image = cv.imread(filename)
    try:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    except:
        print(" --(!) Error reading image ", filename)
        return None
    return image

def filterMatchesByDistance(matches):
    filteredMatches = []
    matches = sorted(matches, key = lambda x:x.distance)
    ptsPairs = min(400, len(matches)*0.3)
    filteredMatches = matches[:ptsPairs]
    return filteredMatches

def filterMatchesRANSAC(matches, keypointsA, keypointsB):
    filteredMatches = []
    if len(matches) >= 4:
        src_pts = np.float32([ keypointsA[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ keypointsB[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 1.0)
        matchesMask = mask.ravel().tolist()

        for i in range(len(matchesMask)):
            if matchesMask[i] == 1:
                filteredMatches.append(matches[i])

    return homography, filteredMatches

def showResult(imgA, keypointsA, imgB, keypointsB, matches, name, homography=None):
    imgMatch = cv.drawMatches(imgA, keypointsA, imgB, keypointsB, matches, None)

    if homography is not None:
        h,w = imgA.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts, homography)
        dst += np.float32([w,0])
        imgMatch = cv.polylines(imgMatch,[np.int32(dst)],True,(0,255,0),3, cv.LINE_AA)

    cv.namedWindow("matches_"+name, cv.WINDOW_KEEPRATIO)
    cv.imshow("matches_"+name, imgMatch)

def extract_local_features(path):
    # 1.a. #
    img = openImage(path)

    height = img.shape[0]
    width = img.shape[1]

    # 1.b.
    fast = cv.FastFeatureDetector_create()

    # 1.c.
    cv.imshow("Before detector",img)
    img_kp = np.zeros((width, height, 1), dtype = "uint8")
    kp = fast.detect(img,None)

    #1.d.
    img_kp = cv.drawKeypoints(img, kp, img_kp, color=(255,0,0))
    cv.imshow("After FAST",img_kp)

    #1.e.
    sift = cv.xfeatures2d.SIFT_create()
    kp = sift.detect(img,None)
    img_kp = cv.drawKeypoints(img, kp, img_kp, color=(255,0,0))
    cv.imshow("After SIFT",img_kp)

    surf = cv.xfeatures2d_SURF.create()
    kp = surf.detect(img,None)
    img_kp = cv.drawKeypoints(img, kp, img_kp, color=(255,0,0))
    cv.imshow("After SURF",img_kp)

    sift = cv.KAZE_create()
    kp = sift.detect(img,None)
    img_kp = cv.drawKeypoints(img, kp, img_kp, color=(255,0,0))
    cv.imshow("After KAZE",img_kp)

def matching_local_features():
    # 2.1.
    detector = cv.xfeatures2d.SIFT_create()

    # 2.b.
    img1 = openImage("images_2/poster_test.jpg")
    kp1, d1 = detector.detectAndCompute(img1,None)

    # 2.c.
    matcher = cv.FlannBasedMatcher()

    for i in range(1,8):
        image = openImage("images_2/poster" + str(i) + ".jpg")
        kp, d = detector.detectAndCompute(image,None)

        matches = matcher.match(d1,d)

        # 2.d.
        showResult(img1,kp1,image,kp,matches,str(i)+"_before_filter")
        
        # 2.e.
        matches_filter_dist = filterMatchesByDistance(matches)
        showResult(img1,kp1,image,kp,matches_filter_dist,str(i)+"_after_filter_dist")

        '''
        # 2.f.
        matches_filter_ransac = filterMatchesRANSAC(matches,kp1,kp)
        showResult(img1,kp1,image,kp,matches_filter_ransac,str(i)+"_after_filter_ransac")
        '''
        
    
def main():

    if len(sys.argv) < 2 or (sys.argv[1] != "1" and sys.argv[1] != "2"):
        print 'python objdetection.py [1=extract_local_features|2=matching_local_features] [filename]'
        return

    if(sys.argv[1] == "1"):
        if(len(sys.argv) != 3):
            print 'python objdetection.py 1 [filename]'
            return
        extract_local_features(sys.argv[2])
    elif(sys.argv[1] == "2"):
        if(len(sys.argv) != 2):
            print 'python objdetection.py 2'
            return
        matching_local_features()
    
    # wait for user input
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
