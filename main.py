import argparse
from src.video_processor import process_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='path to video', required=True)
    args = parser.parse_args()

    process_video(args.video_path)
