# Face Detection CroppingTool for YouTubeShorts

Welcome to the Face Detection CroppingTool for YouTubeShorts project! This innovative Python application leverages the power of OpenCV and MoviePy to perform sophisticated video processing tasks, including dynamic face detection, video cropping, resizing, and audio management. Designed with efficiency and accuracy in mind, this tool ensures that the subject remains in focus, making it ideal for enhancing video content for various applications such as social media, professional presentations, and more.

## Features

- **Dynamic Face Detection**: Utilizes OpenCV's Haar Cascade Classifier to accurately detect faces in video frames, ensuring the subject remains the focal point throughout the video.
- **Adaptive Video Cropping and Resizing**: Automatically adjusts the cropping window based on detected faces, maintaining a 16:9 aspect ratio for optimal viewing experience.
- **Audio Management**: Seamlessly extracts and reattaches audio from the original video to the processed output, ensuring no loss in audio quality or synchronization.
- **Efficient Frame Processing**: Incorporates an intelligent frame update mechanism to optimize processing time without compromising the detection accuracy.

## Customizable Parameters

- **scaleFactor**: Determines the scale factor between image scales for face detection. A higher value might miss smaller faces but improve processing speed.
- **minNeighbors**: The minimum number of neighbors a rectangle should have to be considered a face. Higher values result in fewer detections but with higher quality.
- **check_top_frame**: Number of initial frames to analyze for setting the initial cropping window based on detected faces. Useful for establishing focus at the beginning of the video.
- **update_interval**: The frequency of frame analysis for updating the cropping window. Lower values ensure more frequent updates, providing smoother tracking of the subject.

## Getting Started

### Prerequisites

- Python 3.10
- OpenCV-Python
- MoviePy

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/MichaelChenGithub/YouTubeShorts-CroppingTool.git
   ```
2. Install the required packages:
   ```sh
   pip install opencv-python moviepy
   ```

### Usage

To use this tool, simply specify the input and output video paths in the `main` function and execute the script:
```python
def main():
    input_video_path = 'path/to/your/input/video.mp4'
    output_video_path = 'path/to/your/output/video.mp4'

    processor = VideoProcessor(input_video_path, output_video_path, scaleFactor=1.2, minNeighbors=5, check_top_frame=10, update_interval=5)
    processor.run() # Run the video processing

if __name__ == '__main__':
    main()
```

## How It Works

1. **Initialization**: The `VideoProcessor` class initializes the process by setting up the input and output paths, and preparing the face detection and audio management components.
2. **Face Detection**: For each frame, faces are detected using the Haar Cascade Classifier, and the largest face is used to adapt the cropping window, ensuring the subject remains centered.
3. **Frame Processing**: Frames are cropped and resized according to the updated cropping window, maintaining a consistent focus on the subject.
4. **Audio Management**: The audio track from the original video is extracted and seamlessly integrated into the processed video, ensuring a complete and synchronized final product.
5. **Output**: The processed video is saved to the specified output path, complete with the original audio track and optimized visual content.