def get_overlap_percentage(rect1, rect2):
    """
    Compute the Intersection over Union (IoU) of two rectangles.
    Each rectangle is given as (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = rect1
    x1_p, y1_p, x2_p, y2_p = rect2

    # Compute intersection coordinates
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    # Compute intersection area
    intersection_width = max(0, xi2 - xi1)
    intersection_height = max(0, yi2 - yi1)
    intersection_area = intersection_width * intersection_height

    # Compute area of each rectangle
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)

    # Choose the smaller rectangle's area
    min_area = min(area1, area2)

    # Compute overlap percentage
    overlap_percent = (intersection_area / min_area) * 100

    return overlap_percent

import os

from ultralytics import YOLO
import cv2
model = YOLO(r'C:\Users\mtiurina\source\repos\airo\OtolithOnOtolithCutDetection\runs\detect\train3\weights\best.pt') 
folder = r'C:\Users\mtiurina\source\repos\airo\DataSets\otolith_cuts\test'
threshold = 0.2

if not os.path.exists(os.path.join(folder, "results")):
    os.makedirs(os.path.join(folder, "results"))

for filename in os.listdir(folder):
    if not filename.lower().endswith(".jpg") :
        continue
    img = cv2.imread(os.path.join(folder, filename))
    results = model(img)[0]

    #filter list of boxes
    result_box_data = [box for box in results.boxes.data.tolist() if box[4] > threshold]
    
    #leave only boxes with label 1 (otolith) 
    result_box_data = [box for box in result_box_data if box[5] == 1]
    i=0;
    while i<len(result_box_data):
        found_overlap=False;
        for j in range(i + 1, len(result_box_data)):
           rect1 = result_box_data[i][:4]  
           rect2 = result_box_data[j][:4]
           overlap_percentage = get_overlap_percentage(rect1, rect2)
           if overlap_percentage > 60:
                x1, y1, x2, y2 = rect1
                x1, y1, x2, y2 = rect2
                found_overlap=True;
                overlap_j = j
                break
        if found_overlap:
            #remove the box with the smaller area
            i_box_size = (result_box_data[i][2] - result_box_data[i][0]) * (result_box_data[i][3] - result_box_data[i][1])
            j_box_size = (result_box_data[overlap_j][2] - result_box_data[overlap_j][0]) * (result_box_data[overlap_j][3] - result_box_data[overlap_j][1])
            if i_box_size < j_box_size:
                result_box_data.pop(i)
            else:
                result_box_data.pop(overlap_j)
        else:
            i+=1
    
    #save each otolith image in separate file
    for i in range(len(result_box_data)):
        x1, y1, x2, y2, score, label = result_box_data[i]
        if (label == 1): #otolith 
            cv2.imwrite(os.path.join(folder, "results", filename[:-4] + "_" + str(i) + ".jpg"), img[int(y1):int(y2), int(x1):int(x2)])

    #draw init  image with boxes on it
    for i in range(len(result_box_data)):
        x1, y1, x2, y2, score, label = result_box_data[i]
        if (label == 1): #otolith 
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)
            cv2.putText(img, " {:.2f}".format(score), (int(x1), int(y1 - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 10.3, (0, 255, 0), 10, cv2.LINE_AA)
            
       
            

    cv2.imwrite(os.path.join(folder, "results", filename), img)







