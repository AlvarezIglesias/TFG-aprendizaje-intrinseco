import cv2
import sys
import imagehash
from PIL import Image
import math
import matplotlib.pyplot as plt


engine = {}

total_reward = 0
reward_history = []

def process_frame(frame):
    global total_reward
    img = Image.fromarray(frame).resize((84,84))
    hash = imagehash.ahash(img, hash_size=2)

    if hash in engine:
        engine[hash] += 1
    else:
        engine.setdefault(hash, 1)

    total_reward += 1/math.sqrt(engine[hash])
    reward_history.append(total_reward)



def create_line_graph(values):

    fig, ax = plt.subplots()
    ax.plot(values)
    ax.set(xlabel='X-axis', ylabel='Y-axis', title='Line Graph')
    plt.show()


video_capture = cv2.VideoCapture(sys.argv[1])
if not video_capture.isOpened():
    print("Error opening video file")


success, frame = video_capture.read()

while success:

    process_frame(frame)
    success, frame = video_capture.read()

video_capture.release()

print(reward_history[-1]/len(reward_history))
#create_line_graph(reward_history)