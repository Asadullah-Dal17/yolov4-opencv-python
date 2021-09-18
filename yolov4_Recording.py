import cv2 as cv
import time
Conf_threshold = 0.6
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
# print(class_name)
net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


cap = cv.VideoCapture('pexels-alex-pelsh-6896028.mp4')
frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
# cap.set(cv.CAP_PROP_FPS, 7)
dim = (int(frame_width/4), int(frame_height/4))
print(dim)
out = cv.VideoWriter('OutputVideo3.avi', fourcc, 30.0, dim)
starting_time = time.time()
frame_counter = 0
while True:
    ret, frame = cap.read()

    frame_counter += 1
    if ret == False:
        break

    # if frame_counter == 100:
        # break

    frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_name[classid[0]], score)
        cv.rectangle(frame, box, color, 1)
        # cv.line(frame, (box[0]-3, box[1]-15),
        #         (box[0]+110, box[1]-15), (0, 0, 0), 15)
        cv.rectangle(frame, (box[0]-2, box[1]-20),
                     (box[0]+120, box[1]-4), (100, 130, 100), -1)
        cv.putText(frame, label, (box[0], box[1]-10),
                   cv.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime
    # print(fps)
    cv.line(frame, (18, 43), (140, 43), (0, 0, 0), 27)
    cv.putText(frame, f'FPS: {round(fps,2)}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
    cv.imshow('frame', frame)

    out.write(frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
out.release()

cap.release()
cv.destroyAllWindows()
print('done')
