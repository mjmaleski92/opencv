import numpy as np
import cv2 as cv

shrub_img = cv.imread('Screenshot.jpg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('scrub.jpg', cv.IMREAD_UNCHANGED)

needle_w = needle_img.shape[1]
needle_h = needle_img.shape[0]
result = cv.matchTemplate(shrub_img, needle_img, cv.TM_CCOEFF_NORMED)

# get the best match position
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

result = cv.matchTemplate(shrub_img, needle_img, cv.TM_SQDIFF_NORMED)

print(result)

threshold = 0.09
locations = np.where(result <= threshold)
locations = list(zip(*locations[::-1]))
rectangles = []
for loc in locations:
    rect = [int(loc[0]), int(loc[1]), needle_w, needle_h]
    rectangles.append(rect)
print(rectangles)
if len(rectangles):
    print("Found needle")

    line_color = (0, 255, 0)
    line_type = cv.LINE_4
    thickness = 4

    # need to loop over all the locations and draw the rectangles around them
    for loc in rectangles:
        # determine the box position
        top_left = loc
        bottom_right = (top_left[0]+needle_w, top_left[1]+needle_h)

        # draw the box
        cv.rectangle(shrub_img, tuple(top_left), tuple(bottom_right),
                     line_color, thickness)
cv.imshow('Matches', shrub_img)
cv.waitKey()

'''
threshold = 0.8

if max_val >= threshold:
    print('Found needle')
# get dimensions of the needle image
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    top_left = max_loc
    bottom_right = (top_left[0]+needle_w, top_left[1]+needle_h)

    cv.rectangle(shrub_img, top_left, bottom_right, color=(
        0, 255, 0), thickness=2, lineType=cv.LINE_4)

    cv.imshow('Result', shrub_img)
    cv.waitKey()
else:
    print("Needle not found")
'''
