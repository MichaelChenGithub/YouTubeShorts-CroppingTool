import cv2
import os
from moviepy.editor import VideoFileClip

class AudioManager:
    def __init__(self, input_video_path, output_video_path):
        self.input_video_path = input_video_path
        self.temp_output_video_path = output_video_path.replace('.mp4', '_temp.mp4')  # Temporary file for video without audio
        self.output_video_path = output_video_path

    def add_audio_to_video(self):
        clip = VideoFileClip(self.input_video_path)
        audio = clip.audio
        
        processed_clip = VideoFileClip(self.temp_output_video_path).set_audio(audio)
        processed_clip.write_videofile(self.output_video_path, codec='libx264')
        processed_clip.close()
        audio.close()
        clip.close()
        # 删除临时视频文件
        os.remove(self.temp_output_video_path)

class FaceDetector:
    def __init__(self):
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
        return faces

class VideoProcessor:
    def __init__(self, input_video_path, output_video_path):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path.replace('.mp4', '_temp.mp4')  # Temporary file for video without audio
        self.final_output_path = output_video_path  # Final output file with audio
        self.cap = cv2.VideoCapture(input_video_path)
        self.face_detector = FaceDetector()
        self.audio_manager = AudioManager(input_video_path, self.final_output_path)
        
        self.original_height, self.original_width = self.get_video_dimensions()
        self.crop_height, self.crop_width = self.calculate_crop_size()
        self.set_init_crop_box()

        self.frame_counter = 0  # 添加帧数计数器
        self.update_interval = 5  # 每隔5帧更新一次裁剪框

    def get_video_dimensions(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Can't load video.")
            exit()
        return frame.shape[:2]

    def calculate_crop_size(self):
        crop_height = self.original_height
        crop_width = int(crop_height * 9 / 16)
        if crop_width > self.original_width:
            crop_width = self.original_width
            crop_height = int(crop_width * 16 / 9)
        return crop_height, crop_width
    
    def set_init_crop_box(self):
        self.start_x, self.start_y = (self.original_width - self.crop_width) // 2, (self.original_height - self.crop_height) // 2
        # 檢查前十偵影像的人物偵測結果
        for _ in range(10):
            ret, frame = self.cap.read()
            if not ret:
                return

            faces = self.face_detector.detect_faces(frame)
            if len(faces) > 0:
                fx, fy, fw, fh = self.detect_biggest_face(faces)
                # 更新 start_x 和 start_y 的初始值
                self.update_crop_box(fx, fy, fw, fh)

    def detect_biggest_face(self, faces):
        # 選擇最大的那個臉作為主要臉
        main_face = max(faces, key=lambda box: box[2] * box[3])
        fx, fy, fw, fh = main_face  # 臉部框的座標和尺寸
        return fx, fy, fw, fh

    def update_crop_box(self, fx, fy, fw, fh):
        # 更新裁剪框座標以包含整個臉部框
        self.start_x = max(fx + fw // 2 - self.crop_width // 2, 0)
        self.start_y = max(fy + fh // 2 - self.crop_height // 2, 0)

        # 確保裁剪框不超過畫面邊界
        self.start_x = min(self.start_x, self.original_width - self.crop_width)
        self.start_y = min(self.start_y, self.original_height - self.crop_height)

    def process_frame(self, frame):
        # 只有在特定帧上才检测人脸并更新裁剪框
        if self.frame_counter % self.update_interval == 0:
            faces = self.face_detector.detect_faces(frame)
            if len(faces) > 0:
                fx, fy, fw, fh = self.detect_biggest_face(faces)
                # 计算面部框1/4宽度和高度对应的坐标点
                left_quarter_x = fx + fw // 4
                right_quarter_x = fx + 3 * fw // 4
                top_quarter_y = fy + fh // 4
                bottom_quarter_y = fy + 3 * fh // 4
                # 计算裁切框至少涵蓋3/4臉部
                if (left_quarter_x < self.start_x or right_quarter_x > self.start_x + self.crop_width or
                    top_quarter_y < self.start_y or bottom_quarter_y > self.start_y + self.crop_height):
                    self.update_crop_box(fx, fy, fw, fh)
        self.frame_counter += 1  # 更新帧数计数器
        return frame[self.start_y:self.start_y+self.crop_height, self.start_x:self.start_x+self.crop_width]
    def run(self):
        # 打开视频处理流
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (self.crop_width, self.crop_height))

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            # 处理帧
            processed_frame = self.process_frame(frame)
            # 写入处理后的帧
            out.write(processed_frame)
        # 释放资源
        self.cap.release()
        out.release()
        # 添加音频
        self.audio_manager.add_audio_to_video()
    
def main():
    input_video_path = 'test.mp4'
    output_video_path = 'test_output_automation_per5.mp4'

    # 创建 VideoProcessor 实例
    processor = VideoProcessor(input_video_path, output_video_path)
    processor.run()

if __name__ == '__main__':
    main()