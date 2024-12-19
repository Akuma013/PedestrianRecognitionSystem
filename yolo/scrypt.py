#CORRECTED VERSION

import cv2
import numpy as np
import time

# Load YOLOv4 configuration and weights
config_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\yolov4.cfg'
weights_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\yolov4.weights'
names_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\coco.names'

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# Load class names
with open(names_path, 'r') as f:
    classes = f.read().strip().split('\n')

# Load the video
video_path = 'C:\\Users\\user\\Downloads\\mixkit-busy-street-in-the-city-4000-hd-ready.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize count variables
pedestrians_in_zebra = 0
pedestrians_passed = 0
interval_pedestrians_passed = 0

cars_passed = 0
interval_cars_passed = 0

# Define target classes
pedestrian_class = 'person'
car_class = 'car'

# Define the regions of interest (ROI)
zebra_x1, zebra_y1, zebra_x2, zebra_y2 = 230, 300, 1250, 500  # Zebra crossing region

# Line for car detection (just before zebra crossing)
car_line_y = 480
line_thickness = 2

# Timer variables
start_time = time.time()
interval_duration = 5  # 5-second interval

# Output file
output_file = 'vehicle_and_pedestrian_count.txt'

# For tracking car centers to avoid double counting
previous_car_centers = []

# Function to check if a bounding box fully crosses a given line
def car_passed_line(y, h, line_y):
    return (y + h) >= line_y and y < line_y

# Function to check if a car was counted recently based on center coordinates
def is_new_car(center, previous_centers, threshold=50):
    for prev_center in previous_centers:
        distance = np.linalg.norm(np.array(center) - np.array(prev_center))
        if distance < threshold:
            return False
    return True

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to load.")
        break

    height, width, _ = frame.shape

    # Draw the zebra crossing and car detection line
    cv2.rectangle(frame, (zebra_x1, zebra_y1), (zebra_x2, zebra_y2), (255, 255, 0), 2)
    cv2.line(frame, (zebra_x1, car_line_y), (zebra_x2, car_line_y), (0, 0, 255), line_thickness)

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Perform detection
    detections = net.forward(output_layers)

    # Process each detection
    boxes = []
    confidences = []
    class_ids = []

    current_car_centers = []
    current_pedestrians_in_zebra = 0

    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            center = (x + w // 2, y + h // 2)

            if label == pedestrian_class:
                if zebra_x1 <= center[0] <= zebra_x2 and zebra_y1 <= center[1] <= zebra_y2:
                    current_pedestrians_in_zebra += 1

            if label == car_class:
                if car_passed_line(y, h, car_line_y):
                    if is_new_car(center, previous_car_centers):
                        cars_passed += 1
                        interval_cars_passed += 1
                    current_car_centers.append(center)

            # Draw rectangles around detected objects
            color = (0, 255, 0) if label == pedestrian_class else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Update previous car centers for tracking
    previous_car_centers = current_car_centers

    # Calculate pedestrians passed
    if current_pedestrians_in_zebra > pedestrians_in_zebra:
        pedestrians_passed += (current_pedestrians_in_zebra - pedestrians_in_zebra)
        interval_pedestrians_passed += (current_pedestrians_in_zebra - pedestrians_in_zebra)

    pedestrians_in_zebra = current_pedestrians_in_zebra

    # Display counts on the frame
    cv2.putText(frame, f'Pedestrians Passed: {pedestrians_passed}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Cars Passed: {cars_passed}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Write to file every 5 seconds
    elapsed_time = time.time() - start_time
    if elapsed_time >= interval_duration:
        with open(output_file, 'a') as f:
            f.write(f'Time: {int(elapsed_time)}s - Pedestrians Passed: {interval_pedestrians_passed}, Cars Passed: {interval_cars_passed}\n')
            print(f'Interval: {int(elapsed_time)}s - Pedestrians Passed: {interval_pedestrians_passed}, Cars Passed: {interval_cars_passed}')

        # Reset interval counts and timer
        interval_pedestrians_passed = 0
        interval_cars_passed = 0
        start_time = time.time()

    # Display the frame
    cv2.imshow('Pedestrian and Car Counting', frame)

    # Exit on 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Final output
print(f"Final number of pedestrians passed: {pedestrians_passed}")
print(f"Final number of cars passed: {cars_passed}")










#WARNING NOT COUNTING CORRECTLY CARS AND PEDESTRIANS
# #COUNTING PEDESTRIANS AND CARS WITH CLOCK
# import cv2
# import numpy as np
# import time
#
# # Load YOLOv4 configuration and weights
# config_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\yolov4.cfg'
# weights_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\yolov4.weights'
# names_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\coco.names'
#
# # Load YOLO model
# net = cv2.dnn.readNet(weights_path, config_path)
#
# # Load class names
# with open(names_path, 'r') as f:
#     classes = f.read().strip().split('\n')
#
# # Load the video
# video_path = 'C:\\Users\\user\\Downloads\\mixkit-busy-street-in-the-city-4000-hd-ready.mp4'
# cap = cv2.VideoCapture(video_path)
#
# # Initialize count variables
# pedestrians_in_zebra = 0
# pedestrians_passed = 0
# interval_pedestrians_passed = 0
#
# cars_passed = 0
# interval_cars_passed = 0
#
# # Define target classes
# pedestrian_class = 'person'
# car_class = 'car'
#
# # Define the regions of interest (ROI)
# zebra_x1, zebra_y1, zebra_x2, zebra_y2 = 230, 300, 1250, 500  # Zebra crossing region
#
# # Line for car detection (just before zebra crossing)
# car_line_y = 480
# line_thickness = 2
#
# # Timer variables
# start_time = time.time()
# interval_duration = 5  # 5-second interval
#
# # Output file
# output_file = 'vehicle_and_pedestrian_count.txt'
#
# # Function to check if a bounding box fully crosses a given line
# def car_passed_line(y, h, line_y):
#     return (y + h) >= line_y and y < line_y
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video or failed to load.")
#         break
#
#     height, width, _ = frame.shape
#
#     # Draw the zebra crossing and car detection line
#     cv2.rectangle(frame, (zebra_x1, zebra_y1), (zebra_x2, zebra_y2), (255, 255, 0), 2)
#     cv2.line(frame, (zebra_x1, car_line_y), (zebra_x2, car_line_y), (0, 0, 255), line_thickness)
#
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#
#     # Perform detection
#     detections = net.forward(output_layers)
#
#     # Process each detection
#     boxes = []
#     confidences = []
#     class_ids = []
#
#     for out in detections:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#
#     # Apply non-maxima suppression
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#
#     # Track pedestrians and cars
#     current_pedestrians_in_zebra = 0
#
#     if len(indexes) > 0:
#         for i in indexes.flatten():
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#
#             if label == pedestrian_class:
#                 if zebra_x1 <= (x + w / 2) <= zebra_x2 and zebra_y1 <= (y + h / 2) <= zebra_y2:
#                     current_pedestrians_in_zebra += 1
#
#             if label == car_class:
#                 if car_passed_line(y, h, car_line_y):
#                     cars_passed += 1
#                     interval_cars_passed += 1
#
#             # Draw rectangles around detected objects
#             color = (0, 255, 0) if label == pedestrian_class else (0, 0, 255)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#     # Calculate pedestrians passed
#     if current_pedestrians_in_zebra > pedestrians_in_zebra:
#         pedestrians_passed += (current_pedestrians_in_zebra - pedestrians_in_zebra)
#         interval_pedestrians_passed += (current_pedestrians_in_zebra - pedestrians_in_zebra)
#
#     pedestrians_in_zebra = current_pedestrians_in_zebra
#
#     # Display counts on the frame
#     cv2.putText(frame, f'Pedestrians Passed: {pedestrians_passed}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#     cv2.putText(frame, f'Cars Passed: {cars_passed}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#
#     # Write to file every 5 seconds
#     elapsed_time = time.time() - start_time
#     if elapsed_time >= interval_duration:
#         with open(output_file, 'a') as f:
#             f.write(f'Time: {int(elapsed_time)}s - Pedestrians Passed: {interval_pedestrians_passed}, Cars Passed: {interval_cars_passed}\n')
#             print(f'Interval: {int(elapsed_time)}s - Pedestrians Passed: {interval_pedestrians_passed}, Cars Passed: {interval_cars_passed}')
#
#         # Reset interval counts and timer
#         interval_pedestrians_passed = 0
#         interval_cars_passed = 0
#         start_time = time.time()
#
#     # Display the frame
#     cv2.imshow('Pedestrian and Car Counting', frame)
#
#     # Exit on 'Esc' key
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()
#
# # Final output
# print(f"Final number of pedestrians passed: {pedestrians_passed}")
# print(f"Final number of cars passed: {cars_passed}")








#COUNTING PEDESTRIANS AND CARS WITH CLOCK
# import cv2
# import numpy as np
# import time
#
# # Load YOLOv4 configuration and weights
# config_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\yolov4.cfg'
# weights_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\yolov4.weights'
# names_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\coco.names'
#
# # Load YOLO model
# net = cv2.dnn.readNet(weights_path, config_path)
#
# # Load class names
# with open(names_path, 'r') as f:
#     classes = f.read().strip().split('\n')
#
# # Load the video
# video_path = 'C:\\Users\\user\\Downloads\\mixkit-busy-street-in-the-city-4000-hd-ready.mp4'
# cap = cv2.VideoCapture(video_path)
#
# # Initialize count variables
# pedestrians_in_zebra = 0
# pedestrians_passed = 0
# interval_pedestrians_passed = 0
#
# cars_in_region = 0
# cars_passed = 0
# interval_cars_passed = 0
#
# # Define target classes
# pedestrian_class = 'person'
# car_class = 'car'
#
# # Define the regions of interest (ROI)
# zebra_x1, zebra_y1, zebra_x2, zebra_y2 = 230, 300, 1250, 500  # Zebra crossing
# road_x1, road_y1, road_x2, road_y2 = 230, 500, 1250, 700  # Road region for cars
#
# # Timer variables
# start_time = time.time()
# interval_duration = 5  # 5-second interval
#
# # Output file
# output_file = 'vehicle_and_pedestrian_count.txt'
#
#
# # Function to check if a bounding box is within a given ROI
# def is_in_region(x, y, w, h, region_coords):
#     rx1, ry1, rx2, ry2 = region_coords
#     center_x = x + w / 2
#     center_y = y + h / 2
#     return rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2
#
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video or failed to load.")
#         break
#
#     height, width, _ = frame.shape
#
#     # Draw the zebra crossing and road regions
#     cv2.rectangle(frame, (zebra_x1, zebra_y1), (zebra_x2, zebra_y2), (255, 255, 0), 2)
#     cv2.rectangle(frame, (road_x1, road_y1), (road_x2, road_y2), (0, 0, 255), 2)
#
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#
#     # Perform detection
#     detections = net.forward(output_layers)
#
#     # Process each detection
#     boxes = []
#     confidences = []
#     class_ids = []
#
#     for out in detections:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#
#     # Apply non-maxima suppression
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#
#     # Track pedestrians and cars
#     current_pedestrians_in_zebra = 0
#     current_cars_in_region = 0
#
#     if len(indexes) > 0:
#         for i in indexes.flatten():
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#
#             if label == pedestrian_class:
#                 if is_in_region(x, y, w, h, (zebra_x1, zebra_y1, zebra_x2, zebra_y2)):
#                     current_pedestrians_in_zebra += 1
#
#             if label == car_class:
#                 if is_in_region(x, y, w, h, (road_x1, road_y1, road_x2, road_y2)):
#                     current_cars_in_region += 1
#
#             # Draw rectangles around detected objects
#             color = (0, 255, 0) if label == pedestrian_class else (0, 0, 255)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#     # Calculate pedestrians passed
#     if current_pedestrians_in_zebra > pedestrians_in_zebra:
#         pedestrians_passed += (current_pedestrians_in_zebra - pedestrians_in_zebra)
#         interval_pedestrians_passed += (current_pedestrians_in_zebra - pedestrians_in_zebra)
#
#     pedestrians_in_zebra = current_pedestrians_in_zebra
#
#     # Calculate cars passed
#     if current_cars_in_region > cars_in_region:
#         cars_passed += (current_cars_in_region - cars_in_region)
#         interval_cars_passed += (current_cars_in_region - cars_in_region)
#
#     cars_in_region = current_cars_in_region
#
#     # Display counts on the frame
#     cv2.putText(frame, f'Pedestrians Passed: {pedestrians_passed}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                 (0, 255, 0), 2)
#     cv2.putText(frame, f'Cars Passed: {cars_passed}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#
#     # Write to file every 5 seconds
#     elapsed_time = time.time() - start_time
#     if elapsed_time >= interval_duration:
#         with open(output_file, 'a') as f:
#             f.write(
#                 f'Time: {int(elapsed_time)}s - Pedestrians Passed: {interval_pedestrians_passed}, Cars Passed: {interval_cars_passed}\n')
#             print(
#                 f'Interval: {int(elapsed_time)}s - Pedestrians Passed: {interval_pedestrians_passed}, Cars Passed: {interval_cars_passed}')
#
#         # Reset interval counts and timer
#         interval_pedestrians_passed = 0
#         interval_cars_passed = 0
#         start_time = time.time()
#
#     # Display the frame
#     cv2.imshow('Pedestrian and Car Counting', frame)
#
#     # Exit on 'Esc' key
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()
#
# # Final output
# print(f"Final number of pedestrians passed: {pedestrians_passed}")
# print(f"Final number of cars passed: {cars_passed}")








#ONLY PEDESTRIANS COUNTING ADDED CLOCK ALSO
# import cv2
# import numpy as np
# import time
#
# # Load YOLOv4 configuration and weights
# config_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\yolov4.cfg'
# weights_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\yolov4.weights'
# names_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\coco.names'
#
# # Load YOLO model
# net = cv2.dnn.readNet(weights_path, config_path)
#
# # Load class names
# with open(names_path, 'r') as f:
#     classes = f.read().strip().split('\n')
#
# # Load the video
# video_path = 'C:\\Users\\user\\Downloads\\mixkit-busy-street-in-the-city-4000-hd-ready.mp4'
# cap = cv2.VideoCapture(video_path)
#
# # Initialize count variables
# pedestrians_in_zebra = 0
# pedestrians_passed = 0
# interval_passed = 0  # Count for each 5-second interval
#
# # Define target class for pedestrians
# target_class = 'person'
#
# # Define the zebra crossing ROI as a rectangle (x1, y1, x2, y2)
# zebra_x1, zebra_y1, zebra_x2, zebra_y2 = 230, 300, 1250, 500
#
# # Timer variables
# start_time = time.time()
# interval_duration = 5  # 5-second interval
#
# # Open the output file
# output_file = 'pedestrian_count.txt'
#
#
# # Function to check if a bounding box is within the zebra crossing ROI
# def is_in_zebra_crossing(x, y, w, h, zebra_coords):
#     zx1, zy1, zx2, zy2 = zebra_coords
#     center_x = x + w / 2
#     center_y = y + h / 2
#     return zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2
#
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video or failed to load.")
#         break
#
#     height, width, channels = frame.shape
#
#     # Draw the zebra crossing ROI as a rectangle
#     cv2.rectangle(frame, (zebra_x1, zebra_y1), (zebra_x2, zebra_y2), (255, 255, 0), 2)
#
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#
#     # Perform detection
#     detections = net.forward(output_layers)
#
#     # Process each detection
#     boxes = []
#     confidences = []
#     class_ids = []
#
#     for out in detections:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#
#     # Apply non-maxima suppression
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#
#     # Track pedestrians in the zebra crossing
#     current_pedestrians_in_zebra = 0
#
#     if len(indexes) > 0:
#         for i in indexes.flatten():
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#
#             if label == target_class:
#                 if is_in_zebra_crossing(x, y, w, h, (zebra_x1, zebra_y1, zebra_x2, zebra_y2)):
#                     current_pedestrians_in_zebra += 1
#
#                 # Draw rectangle around detected pedestrian
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#     # Calculate the number of pedestrians passed
#     if current_pedestrians_in_zebra > pedestrians_in_zebra:
#         pedestrians_passed += (current_pedestrians_in_zebra - pedestrians_in_zebra)
#         interval_passed += (current_pedestrians_in_zebra - pedestrians_in_zebra)
#
#     pedestrians_in_zebra = current_pedestrians_in_zebra
#
#     # Display counts on the frame
#     cv2.putText(frame, f'Pedestrians on Zebra: {pedestrians_in_zebra}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                 (0, 255, 0), 2)
#     cv2.putText(frame, f'Pedestrians Passed: {pedestrians_passed}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                 (0, 255, 0), 2)
#
#     # Write to file every 5 seconds
#     elapsed_time = time.time() - start_time
#     if elapsed_time >= interval_duration:
#         with open(output_file, 'a') as f:
#             f.write(f'Time: {int(elapsed_time)}s - Pedestrians Passed: {interval_passed}\n')
#             print(f'Interval: {int(elapsed_time)}s - Pedestrians Passed: {interval_passed}')
#
#         # Reset the interval count and timer
#         interval_passed = 0
#         start_time = time.time()
#
#     # Display the frame
#     cv2.imshow('Pedestrian Counting on Zebra Crossing', frame)
#
#     # Exit on 'Esc' key
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()
#
# # Final output
# print(f"Final number of pedestrians passed: {pedestrians_passed}")






# import cv2
# import numpy as np
#
# # Load YOLOv4 configuration and weights
# config_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\yolov4.cfg'
# weights_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\yolov4.weights'
# names_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\coco.names'
#
# # Load YOLO model
# net = cv2.dnn.readNet(weights_path, config_path)
#
# # Load class names
# with open(names_path, 'r') as f:
#     classes = f.read().strip().split('\n')
#
# # Load the video
# # video_path = 'C:\\Users\\user\\Downloads\\mixkit-traffic-in-a-crossroads-4343-hd-ready.mp4'
# video_path = 'C:\\Users\\user\\Downloads\\mixkit-busy-street-in-the-city-4000-hd-ready.mp4'
# cap = cv2.VideoCapture(video_path)
#
# # Initialize count variables
# pedestrians_in_zebra = 0
# pedestrians_passed = 0
#
# # Define target class for pedestrians
# target_class = 'person'
#
# # Define the zebra crossing ROI as a rectangle (x1, y1, x2, y2)
# # Adjust these values according to the video
# zebra_x1, zebra_y1, zebra_x2, zebra_y2 = 230, 300, 1250, 500
# # zebra_x1, zebra_y1, zebra_x2, zebra_y2 = 300, 300, 1000, 160
#
# # Function to check if a bounding box is within the zebra crossing ROI
# def is_in_zebra_crossing(x, y, w, h, zebra_coords):
#     zx1, zy1, zx2, zy2 = zebra_coords
#     # Calculate the center of the bounding box
#     center_x = x + w / 2
#     center_y = y + h / 2
#     # Check if the center is inside the zebra crossing rectangle
#     return zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2
#
#
# while True:
#     # Read frame from the video
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video or failed to load.")
#         break
#
#     height, width, channels = frame.shape
#
#     # Draw the zebra crossing ROI as a rectangle
#     cv2.rectangle(frame, (zebra_x1, zebra_y1), (zebra_x2, zebra_y2), (255, 255, 0), 2)
#
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#
#     # Perform detection
#     detections = net.forward(output_layers)
#
#     # Process each detection
#     boxes = []
#     confidences = []
#     class_ids = []
#
#     for out in detections:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:  # Threshold for detection
#                 # Object detected
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#
#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#
#     # Apply non-maxima suppression to reduce overlapping boxes
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#
#     # Track pedestrians in the zebra crossing
#     current_pedestrians_in_zebra = 0
#
#     if len(indexes) > 0:
#         for i in indexes.flatten():
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#
#             # If the detected object is a pedestrian
#             if label == target_class:
#                 # Check if the pedestrian is within the zebra crossing
#                 if is_in_zebra_crossing(x, y, w, h, (zebra_x1, zebra_y1, zebra_x2, zebra_y2)):
#                     current_pedestrians_in_zebra += 1
#
#                 # Draw rectangle for the detected pedestrian
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#     # Update the count of pedestrians who passed through the zebra crossing
#     if current_pedestrians_in_zebra > pedestrians_in_zebra:
#         pedestrians_passed += (current_pedestrians_in_zebra - pedestrians_in_zebra)
#
#     pedestrians_in_zebra = current_pedestrians_in_zebra
#
#     # Display the count on the frame
#     cv2.putText(frame, f'Pedestrians on Zebra: {pedestrians_in_zebra}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#     cv2.putText(frame, f'Pedestrians Passed: {pedestrians_passed}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#     # Display the frame
#     cv2.imshow('Pedestrian Counting on Zebra Crossing', frame)
#
#     # Stop if 'Esc' key is pressed
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# # Release video and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
#
# # Print the final count of pedestrians who passed
# print(f"Final number of pedestrians passed: {pedestrians_passed}")







# import cv2
# import numpy as np
#
# # Load YOLOv4 configuration and weights
# config_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\yolov4.cfg'
# weights_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\yolov4.weights'
# names_path = 'C:\\Users\\user\\Desktop\\UFAR\\Grade2\\Project S1\\TrafficRec\\yolo\\coco.names'
#
# # Load YOLO model
# net = cv2.dnn.readNet(weights_path, config_path)
#
# # Load class names
# with open(names_path, 'r') as f:
#     classes = f.read().strip().split('\n')
#
# # Load the video
# video_path = 'C:\\Users\\user\\Downloads\\mixkit-busy-street-in-the-city-4000-hd-ready.mp4'
# cap = cv2.VideoCapture(video_path)
#
# # Initialize count dictionary for the specified classes (vehicles and pedestrians)
# target_classes = ['car', 'bus', 'truck', 'motorbike', 'bicycle', 'person']
# object_count = {cls: 0 for cls in target_classes}
#
# # List to store detected bounding boxes (for counting)
# previous_detections = []
#
#
# def is_new_detection(new_bbox, prev_bboxes, threshold=50):
#     for prev_bbox in prev_bboxes:
#         x1, y1, w1, h1 = prev_bbox
#         x2, y2, w2, h2 = new_bbox
#
#         # Calculate distance between center points of the bounding boxes
#         center1 = (x1 + w1 / 2, y1 + h1 / 2)
#         center2 = (x2 + w2 / 2, y2 + h2 / 2)
#
#         distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
#
#         # If the distance is smaller than the threshold, consider it the same object
#         if distance < threshold:
#             return False
#     return True
#
#
# while True:
#     # Read frame from the video
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video or failed to load.")
#         break
#
#     height, width, channels = frame.shape
#
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#
#     # Perform detection
#     detections = net.forward(output_layers)
#
#     # Process each detection
#     boxes = []
#     confidences = []
#     class_ids = []
#
#     for out in detections:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:  # Threshold for detection
#                 # Object detected
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#
#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#
#     # Apply non-maxima suppression to reduce overlapping boxes
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#
#     # List to store the bounding boxes of this frame
#     current_frame_bboxes = []
#
#     # Draw the boxes on the frame
#     if len(indexes) > 0:
#         for i in indexes.flatten():
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#
#             # Check if the label is in the target classes (vehicles and pedestrians)
#             if label in target_classes:
#                 # Check if the new bounding box is a new detection
#                 if is_new_detection([x, y, w, h], previous_detections):
#                     object_count[label] += 1
#                     current_frame_bboxes.append([x, y, w, h])
#
#                 # Draw rectangle for the detected object
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if label == "person" else (0, 0, 255), 2)
#                 cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#     # Add the current detections to the previous detections memory
#     previous_detections.extend(current_frame_bboxes)
#
#     # Display the frame
#     cv2.imshow('Object Detection', frame)
#
#     # Stop if 'Esc' key is pressed
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# # Release video and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
#
# # Print the final count of detected objects
# print("Detection summary:")
# for obj, count in object_count.items():
#     print(f"{obj}: {count}")
