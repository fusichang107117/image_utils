import cv2


def get_frames_by_second(video_path, start, end, interval, width, height):
    cap = cv2.VideoCapture(video_path)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    fps = 0
    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)

    start_frame_id = int(start * fps)
    end_frame_id = int(end * fps)
    interval_frame = int(interval * fps)
    frames = []
    frame_id = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if frame_id >= start_frame_id and frame_id <= end_frame_id:
            if (frame_id-start_frame_id)%interval_frame == 0:
                frame = cv2.resize(frame, (width, height))
                frames.append(frame)
        frame_id = frame_id + 1
    cap.release()
    return frames

def get_frames_by_frame(video_path, start, end, interval, width, height):
    cap = cv2.VideoCapture(video_path)

    frames = []
    frame_id = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if frame_id >= start and frame_id <= end:
            if (frame_id-start)%interval == 0:
                frame = cv2.resize(frame, (width, height))
                frames.append(frame)
        frame_id = frame_id + 1
    cap.release()
    return frames


if __name__ == '__main__' :
    frames = get_frames_by_second('/home/leon/Download/toystory.mp4',start=10.0, end=15.0, interval=0.3, width=640, height=360)
    for frame in frames:
        cv2.imshow('image', frame)
        k = cv2.waitKey(0)
        # q exit
        if (k & 0xff == ord('q')):
            break
    cv2.destroyAllWindows()

    frames = get_frames_by_frame('/home/leon/Download/toystory.mp4', start=1000.0, end=1500.0, interval=100, width=1280, height=720)
    for frame in frames:
        cv2.imshow('image', frame)
        k = cv2.waitKey(0)
        # q exit
        if (k & 0xff == ord('q')):
            break
    cv2.destroyAllWindows()