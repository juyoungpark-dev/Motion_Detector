# Juyoung Park
# 1/24/2021
# Motion Detector Program in Python
# resources: Udemy Course - The Python Mega Course: Build 10 Real World Applications
# https://www.udemy.com/course/t%20he-python-mega-course/learn/lecture/4775502?start=855#questions

# OpenCV: open-source library for the "Computer Vision"
# used for machine learning, image processing
# it can process images, videos to identify objects
# usages: face/object recognition; surveillance;
# count #of people or vehicles along with speed; etc;

# pip install opencv-python
# if you run the py program, camera will be operated
# press 'q' to quit

import cv2,time,pandas
from datetime import datetime

first_frame = None
status_list = [None,None]
times = []
df=pandas.DataFrame(columns=["Start","End"])


video = cv2.VideoCapture(0)

while(True):
    # python captures the first frame
    check, frame = video.read()  # returns true or false
    status = 0
    # converts to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurs the frame
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    # calculates the difference
    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # helps in extracting the contours from the binary image
    # use a copy of the image(.copy()) since findContours alters the image
    # cv2.findContours(sourceImage,contour_retrieval_mode,contour_approximation_method)

    (cnts, _) = cv2.findContours(thresh_frame.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    status_list.append(status)

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    # display the resulting frame
    cv2.imshow('Gray Frame', gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Fraame", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        if status==1:
            times.append(datetime.now())
        break

# outside of while loop:
print(status_list)
print(times)

for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)
df.to_csv("Times.csv")

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
