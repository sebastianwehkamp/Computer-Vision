import numpy as np
import cv2
import copy

def matchKeypoints(kpsA, kpsB, fA, fB):      
        # Match using knn
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(fA, fB, 2)
        matches = []

		# loop over the raw matches
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                matches.append((m[0].trainIdx, m[0].queryIdx))

		# At least 4 matches
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
			
            # Use RANSAC
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,4.0)
            
            return (matches, H, status)

        return None

def drawMatches(imgA, imgB, kpsA, kpsB, matches, status):
		# Create output image with correct size and 2 images
		(hA, wA) = imgA.shape[:2]
		(hB, wB) = imgB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imgA
		vis[0:hB, wA:] = imgB

		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully matched
			if s == 1:
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
		return vis

def drawFeatures(img, kps):
    vis = copy.deepcopy(img)
    for (x,y) in kps:
        cv2.circle(vis,(x,y),2,(0,0,255),2)
    return vis

def getFeatures(img):
  	# Get SIFT features
  	descriptor = cv2.xfeatures2d.SIFT_create()
  	(kps, features) = descriptor.detectAndCompute(img, None)
  	kps = np.float32([kp.pt for kp in kps])
  	return (kps, features)


img1 = cv2.imread('moon2.png', 1)
img2 = cv2.imread('moon1.png', 1)

(KeypointsA, featuresA) = getFeatures(img1)
(KeypointsB, featuresB) = getFeatures(img2)

F1 = drawFeatures(img1,KeypointsA)
F2 = drawFeatures(img2,KeypointsB)

(matches, H, status) = matchKeypoints(KeypointsA, KeypointsB, featuresA, featuresB)
vis = drawMatches(img1, img2, KeypointsA, KeypointsB, matches,status)

result = cv2.warpPerspective(img1, H,(img1.shape[1] + img2.shape[1], img1.shape[0]))
result[0:img2.shape[0], 0:img2.shape[1]] = img2

cv2.imwrite( "F1.png", F1 )
cv2.imwrite( "F2.png", F2 )

cv2.imwrite('vis.png',vis)
cv2.imwrite('result.png',result)
