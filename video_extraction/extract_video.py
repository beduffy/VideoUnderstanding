import preprocess_video
import filmstrip


def extract_video(video_path):
    video_name = video_path.split('/')[-1]

    preprocess_video.extractFrames(video_path, video_path, video_name)