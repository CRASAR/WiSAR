# Importing all necessary libraries
import cv2
import os

def vid2frames(in_file_path, out_folder_path, frame_prefix=None, frame_format="jpg", fps_sample_rate=1):
    frame_format_clean = frame_format.replace(" ", "").replace(".", "")
    frame_prefix_clean = "" if frame_prefix is None else str(frame_prefix)

    in_file_name_ext = os.path.split(in_file_path)[-1]
    in_file_name, ext = in_file_name_ext.split(".")
    print("Found file:", in_file_name_ext)

    # Read the video from specified path
    cam = cv2.VideoCapture(in_file_path)

    fps = 1
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = cam.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Found a video with", fps, "frames/sec")
    else :
        fps = cam.get(cv2.CAP_PROP_FPS)
        print("Found a video with", fps, "frames/sec")

    sample_rate = round(fps/fps_sample_rate)
    print("Will sample 1 frame for every", sample_rate, "frames.")
 
    try:
        # creating a folder named data
        if not os.path.exists(out_folder_path):
            os.makedirs(out_folder_path)

    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')

    # frame
    currentframe = 0

    frames_to_write = True
    while(frames_to_write):
          
        # reading from frame
        ret,frame = cam.read()

        if ret:
            if(currentframe % sample_rate == 0):
                # if video is still left continue creating images
                file_name = frame_prefix_clean + in_file_name + "_" + ext + "_" + str(currentframe) + "." + frame_format_clean
                name = os.path.join(out_folder_path, file_name)

                # writing the extracted images
                cv2.imwrite(name, frame)

                print ('Created... ' + name)

            currentframe += 1
        else:
            frames_to_write = False

    # Release all space and windows once done
    cam.release()

if(__name__ == "__main__"):
    import argparse

    parser = argparse.ArgumentParser(description='Convert a video to its respective frames.')
    parser.add_argument('--in_file_path', type=str, help='The path to the folder that needs to be read.')
    parser.add_argument('--out_folder_path', type=str, help='The path to the folder where all the frames should be saved to.')
    parser.add_argument('--frame_prefix', type=str, help='The prefix to each frame.', default="")
    parser.add_argument('--frame_format', type=str, help='The image format that should be used to save the frames.', default="jpg")
    parser.add_argument('--fps_sample_rate', type=int, help='How many frames per second should be sampled.', default=2)
    args = parser.parse_args()

    vid2frames(args.in_file_path, args.out_folder_path, args.frame_prefix, args.frame_format, args.fps_sample_rate)