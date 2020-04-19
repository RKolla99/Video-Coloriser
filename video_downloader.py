import json
from pytube import YouTube
import cv2
import numpy as np
import h5py

with open('videodatainfo_2017.json') as f:
    videoData = json.load(f)

path = 'dataset/color'
grayPath = 'dataset/grayscale'
rgbVid = []
grayVid = []
videoCounter = 0
categoryList = [1, 1, 6, 6, 7, 11, 11, 13, 14, 14]
for video in videoData['videos']:
    if video['category'] in categoryList:
        try:
            yt = YouTube(video['url'])
        except:
            print('Connection Error')

        videoFile = yt.streams.filter(file_extension='mp4').first()
        print(videoFile)
        # yt.set_filename(f'video_{videoCounter}')

        try:
            videoFile.download(path, filename=f'video_{videoCounter}')
            rgbFrames = np.empty((12, 256, 256, 3), np.dtype('uint8'))
            grayFrames = np.empty((12, 256, 256), np.dtype('uint8'))

            cap = cv2.VideoCapture(f'{path}/video_{videoCounter}.mp4')
            fps = cap.get(cv2.CAP_PROP_FPS)
            frameCount = fps * 120
            batchCount = 0

            ret, frame = cap.read()
            while cap.isOpened() and frameCount // fps <= 180:
                if not ret:
                    break

                ret, frame = cap.read()

                indice = int(batchCount % 12)
                rgbFrames[indice] = cv2.resize(frame, (256, 256))
                grayFrames[indice] = cv2.cvtColor(
                    rgbFrames[indice], cv2.COLOR_BGR2GRAY)

                if indice == 0:
                    rgbVid.append(rgbFrames)
                    grayVid.append(grayFrames)
                    rgbFrames = np.empty((12, 256, 256, 3), np.dtype('uint8'))
                    grayFrames = np.empty((12, 256, 256), np.dtype('uint8'))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                batchCount += 1
                frameCount += 1

            cap.release()
            videoCounter += 1
            categoryList.remove(video['category'])
        except Exception as e:
            print(e)
            print('Error while downloading')

    if not categoryList:
        break

rgbFile = h5py.File('./dataset/rgbVideos.h5', 'w')
grayFile = h5py.File('./dataset/grayVideos.h5', 'w')

rgbFile.create_dataset('videos', data=rgbVid, compression='gzip', chunks=True)
grayFile.create_dataset('videos', data=grayVid,
                        compression='gzip', chunks=True)
