import cv2
import os

def get_frames_by_second(video_path, start = 0, end = 0, interval = 0, width = 0, height = 0):
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
                if width != 0 and height != 0:
                    frame = cv2.resize(frame, (width, height))
                frames.append(frame)
        frame_id = frame_id + 1
    cap.release()
    return frames

def get_frames_by_frame(video_path, start = 0, end = 0, interval = 0, width = 0, height = 0):
    cap = cv2.VideoCapture(video_path)

    frames = []
    frame_id = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if frame_id >= start and frame_id <= end:
            if (frame_id-start)%interval == 0:
                if width != 0 and height != 0:
                    frame = cv2.resize(frame, (width, height))
                frames.append(frame)
        frame_id = frame_id + 1
    cap.release()
    return frames


def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.mp4':
                L.append(os.path.join(root, file))
    return L

def get_frames_to_files(video_dir, out_dir, start, end, interval, type):
    out_dir_name = "xiaomi_frames_%s_%d_%d_%d"%(type,start,end,interval)
    out_dir = os.path.join(out_dir,out_dir_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    videos = file_name(video_dir)

    file_list_path = os.path.join(video_dir,"vidoe_list.txt")
    f_list = open(file_list_path,"w")

    video_id = 0
    for video in videos:
        print video_id, video
        f_list.write(video + "\n")

        video_name = os.path.basename(video)
        video_name = video_name.split(".")[0]
        img_out_dir = os.path.join(out_dir,video_name)
        if not os.path.exists(img_out_dir):
            os.makedirs(img_out_dir)

        frames = []
        if type == "second":
            frames = get_frames_by_second(video,start,end,interval)
        elif type == "frame":
            frames = get_frames_by_frame(video, start, end, interval)

        frame_id = 0
        for frame in frames:
            img_path = os.path.join(img_out_dir,str(frame_id)+".jpg")
            cv2.imwrite(img_path,frame)
            frame_id = frame_id + 1

        video_id = video_id + 1

    f_list.close()

if __name__ == '__main__' :
    get_frames_to_files("/home/leon/Images/xiaomi_video","/home/leon/Images/", 0,5,1,"second")
    # frames = get_frames_by_second('/home/leon/Download/toystory.mp4',start=10.0, end=15.0, interval=0.3, width=640, height=360)
    # for frame in frames:
    #     cv2.imshow('image', frame)
    #     k = cv2.waitKey(0)
    #     # q exit
    #     if (k & 0xff == ord('q')):
    #         break
    # cv2.destroyAllWindows()
    #
    # frames = get_frames_by_frame('/home/leon/Download/toystory.mp4', start=1000.0, end=1500.0, interval=100, width=1280, height=720)
    # for frame in frames:
    #     cv2.imshow('image', frame)
    #     k = cv2.waitKey(0)
    #     # q exit
    #     if (k & 0xff == ord('q')):
    #         break
    # cv2.destroyAllWindows()
