import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import detection_Licence

 image = np.invert(detection_Licence.img)
#the starting portion of the Image segmentation i.e. conversion to GRAY and Binarisation
car_image = imutils.rotate(image, 270)
gray_car_image = car_image * 255
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(gray_car_image, cmap="gray")
threshold_value = threshold_otsu(gray_car_image)
license_plate = gray_car_image > threshold_value

labelled_plate = measure.label(license_plate)

fig, ax1 = plt.subplots(1)
ax1.imshow(license_plate, cmap="gray")

# dimension elimination as per our assumption

character_dimensions = (0.30*license_plate.shape[0], 0.70*license_plate.shape[0], 0.05*license_plate.shape[1], 0.2*license_plate.shape[1])
min_height, max_height, min_width, max_width = character_dimensions

Rec_char = []
col_lis=[]
for regions in regionprops(labelled_plate):
    y0, x0, y1, x1 = regions.bbox
    region_height = y1 - y0
    region_width = x1 - x0

    if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
        roi = license_plate[y0:y1, x0:x1]

        # draw rectangle over the character detected
        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="yellow",
                                       linewidth=2, fill=False)
        ax1.add_patch(rect_border)

        # resize the characters to 20 X 20 
        resized_char = resize(roi, (20, 20))
        Rec_char.append(resized_char)
        col_lis.append(x0)
plt.show()
