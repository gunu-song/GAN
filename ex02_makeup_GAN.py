import dlib
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import numpy as np
import cv2

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')
predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
index = list(range(0, 68))

def preprocess(img):
    return img.astype(np.float32) / 127.5 - 1

def postprocess(img):
    return((img+1.) * 127.5).astype(np.uint8)

def align_faces(img):
    dets = detector(img, 1) # 1채널

    objs = dlib.full_object_detections()

    for detection in dets:
        s = predictor(img, detection)
        objs.append(s)
    faces = dlib.get_face_chips(img, objs, size=256, padding=1)
    return faces

def handle_close(evt):
    print('close figure!')
    cap.release()

def beautyGAN(Orign_image):

    print("Success Detection")
    Orign_image_face = align_faces(Orign_image)

    Makeup_image = cv2.imread('./images/makeup/vFG56.png', cv2.IMREAD_COLOR)
    Makeup_image = cv2.cvtColor(Makeup_image, cv2.COLOR_BGR2RGB)
    Makeup_image_face = align_faces(Makeup_image)

    src_img = Orign_image_face[0]
    ref_img = Makeup_image_face[0]

    X_img = preprocess(src_img)
    X_img = np.expand_dims(X_img, axis=0)

    Y_img = preprocess(ref_img)
    Y_img = np.expand_dims(Y_img, axis=0)

    output = sess.run(Xs, feed_dict={
        X: X_img,
        Y: Y_img
    })
    output_img = postprocess(output[0])

    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    return output_img

#### BeautyGAN
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
saver = tf.compat.v1.train.import_meta_graph('./models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('models'))
graph = tf.compat.v1.get_default_graph()
X = graph.get_tensor_by_name('X:0')  # source
#
#
Y = graph.get_tensor_by_name('Y:0')  # reference
Xs = graph.get_tensor_by_name('generator/xs:0')  # output
#### BeautyGAN end

## webcam use
cap = cv2.VideoCapture(1)

while True:
    face_detected = False  # True 일 때 GAN 실행
    retval, frame = cap.read()
    height, width = frame.shape[:2]
    print('width : ', width, ', height : ', height)

    ### 얼굴 영역 인식 확인 ###
    dets = detector(frame, 1)   # 1채널
    for face in dets:
        if dets != 0:
            face_detected = True  # 얼굴 인식 시 face_detected 가 True로 변함
        else:
            print('cannot find faces!')
    ### 얼굴 영역 인식 끝 ###
    if not retval:
        break

    Origin_image = frame.copy()
    #Origin_image = cv2.imread('./images/11.jpg',cv2.IMREAD_COLOR)
    #face_detected = True  # 얼굴 인식 시 face_detected 가 True로 변함

    if face_detected == True:  # face_detected 가 True 일 때만 GAN 실행
        Origin_image = beautyGAN(Origin_image)

    cv2.imshow('facial landmarks', Origin_image)

    key = cv2.waitKey(20)
    if key == 27:
        if cap.isOpened():
            cap.release()
        break