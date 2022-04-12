from Detector import *
import os

def main():
    #videoPath = "test_videos/4K_Jan1902_Halifax_England.mp4"
    videoPath = "test_videos/drone1.MOV"

    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()

if  __name__ == '__main__':
    main()