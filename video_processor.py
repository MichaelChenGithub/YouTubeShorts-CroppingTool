import cv2
import os
from moviepy.editor import VideoFileClip

class AudioManager:
    """
    負責處理影片的音訊。
    能夠從原始影片中提取音訊，並將其添加到經過處理（例如裁剪、調整大小等）後的影片中。
    """
    def __init__(self, input_video_path, output_video_path):
        """
        初始化 AudioManager。

        Parameters:
        - input_video_path: 原始影片的路徑。
        - output_video_path: 處理後將要儲存的影片路徑。
        """
        self.input_video_path = input_video_path
        self.temp_output_video_path = output_video_path.replace('.mp4', '_temp.mp4')  # Temporary file for video without audio
        self.output_video_path = output_video_path

    def add_audio_to_video(self):
        """
        從原始影片提取音訊，並將其添加到處理後的影片中。
        """
        # 加載原始影片並提取音訊
        clip = VideoFileClip(self.input_video_path)
        audio = clip.audio

        # 加載處理後的影片並設定其音訊軌道
        processed_clip = VideoFileClip(self.temp_output_video_path).set_audio(audio)
        processed_clip.write_videofile(self.output_video_path, codec='libx264')
        processed_clip.close()
        audio.close()
        clip.close()

        # 刪除臨時無音訊影片檔案
        os.remove(self.temp_output_video_path)

class FaceDetector:
    """
    用於在影片中檢測人臉的類別。
    使用 OpenCV 的 Haar Cascade 分類器來識別人臉。
    """
    def __init__(self):
        """
        初始化 FaceDetector，加載用於人臉檢測的 Haar Cascade 模型。
        """
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

    def detect_faces(self, frame):
        """
        在給定的影像幀中檢測人臉。

        Parameters:
        - frame: 影像幀（來自影片的單幀）。

        Returns:
        - faces: 檢測到的人臉列表，每個人臉由一個矩形框表示（x, y, w, h）。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
        return faces

class VideoProcessor:
    """
    處理影片的主要類別，包括裁剪、調整大小並保持目標（如人臉）在畫面中心。
    """
    def __init__(self, input_video_path, output_video_path):
        """
        初始化 VideoProcessor。

        Parameters:
        - input_video_path: 原始影片的路徑。
        - output_video_path: 處理後影片的儲存路徑。
        """
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path.replace('.mp4', '_temp.mp4')  # 生成一個臨時檔案儲存無音訊影片
        self.final_output_path = output_video_path  # 最終包含音訊的影片儲存路徑
        self.cap = cv2.VideoCapture(input_video_path) # 打開原始影片
        self.face_detector = FaceDetector() # 創建人臉檢測器實例
        self.audio_manager = AudioManager(input_video_path, self.final_output_path) # 創建音訊管理器實例
        
        # 獲取影片的尺寸並計算裁剪大小
        self.original_height, self.original_width = self.get_video_dimensions()
        self.crop_height, self.crop_width = self.calculate_crop_size()
        self.set_init_crop_box()

        self.frame_counter = 0  # 添加帧数计数器，用于控制裁剪框的更新频率
        self.update_interval = 5  # 每隔5帧更新一次裁剪框

    def get_video_dimensions(self):
        """
        獲取影片的尺寸（寬度和高度）。

        Returns:
        - (height, width): 影片的高度和寬度。
        """
        ret, frame = self.cap.read() # 讀取一幀以獲取尺寸
        if not ret:
            print("Can't load video.")
            exit()
        return frame.shape[:2]

    def calculate_crop_size(self):
        """
        計算裁剪框的大小，保持16:9的寬高比，並確保裁剪框不會超過原始影片的尺寸。

        Returns:
        - (crop_height, crop_width): 裁剪框的高度和寬度。
        """
        crop_height = self.original_height
        crop_width = int(crop_height * 9 / 16) # 按16:9比例計算寬度
        if crop_width > self.original_width:
            crop_width = self.original_width
            crop_height = int(crop_width * 16 / 9) # 如果計算的寬度超過原始寬度，則調整高度
        return crop_height, crop_width
    
    def set_init_crop_box(self):
        """
        設置初始裁剪框的位置，並嘗試根據影片前幾帧中檢測到的最大人臉來調整裁剪框的位置。
        """
        self.start_x, self.start_y = (self.original_width - self.crop_width) // 2, (self.original_height - self.crop_height) // 2
        for _ in range(10): # 檢查前十帧
            ret, frame = self.cap.read()
            if not ret:
                return

            faces = self.face_detector.detect_faces(frame)
            if len(faces) > 0:
                fx, fy, fw, fh = self.detect_biggest_face(faces)
                # 更新 start_x 和 start_y 的初始值
                self.update_crop_box(fx, fy, fw, fh) # 根据检测到的最大人脸调整裁剪框位置

    def detect_biggest_face(self, faces):
        """
        從檢測到的人臉中選擇面積最大的一個。

        Parameters:
        - faces: 檢測到的人臉列表，每個人臉由一個矩形框表示（x, y, w, h）。

        Returns:
        - (fx, fy, fw, fh): 面積最大人臉的矩形框座標和尺寸。
        """
        main_face = max(faces, key=lambda box: box[2] * box[3]) # 選擇面積最大的人臉
        fx, fy, fw, fh = main_face  # 獲取座標和尺寸
        return fx, fy, fw, fh

    def update_crop_box(self, fx, fy, fw, fh):
        """
        根據檢測到的人臉位置更新裁剪框的位置，確保人臉位於裁剪框內。

        Parameters:
        - fx, fy, fw, fh: 人臉的矩形框座標和尺寸。
        """
        # 更新裁剪框的起始坐標，使其以人臉為中心
        self.start_x = max(fx + fw // 2 - self.crop_width // 2, 0)
        self.start_y = max(fy + fh // 2 - self.crop_height // 2, 0)

        # 確保裁剪框不超過畫面邊緣
        self.start_x = min(self.start_x, self.original_width - self.crop_width)
        self.start_y = min(self.start_y, self.original_height - self.crop_height)

    def process_frame(self, frame):
        """
        處理單幀影像，包括檢測人臉、更新裁剪框以及裁剪影像。

        Parameters:
        - frame: 從影片中讀取的單幀影像。

        Returns:
        - 裁剪後的影像幀。
        """
        # 每隔特定幀數檢測一次人臉並可能更新裁剪框
        if self.frame_counter % self.update_interval == 0:
            faces = self.face_detector.detect_faces(frame)
            if len(faces) > 0:
                fx, fy, fw, fh = self.detect_biggest_face(faces)
                # 計算臉部的四分之一點座標，以決定是否需要移動裁剪框
                left_quarter_x = fx + fw // 4
                right_quarter_x = fx + 3 * fw // 4
                top_quarter_y = fy + fh // 4
                bottom_quarter_y = fy + 3 * fh // 4
                # 若臉部的四分之一點座標超出當前裁剪框，則更新裁剪框位置
                if (left_quarter_x < self.start_x or right_quarter_x > self.start_x + self.crop_width or
                    top_quarter_y < self.start_y or bottom_quarter_y > self.start_y + self.crop_height):
                    self.update_crop_box(fx, fy, fw, fh)
        self.frame_counter += 1  # 更新幀計數器
        return frame[self.start_y:self.start_y+self.crop_height, self.start_x:self.start_x+self.crop_width] # 裁剪並返回當前幀
    def run(self):
        """
        執行影片處理流程，包括讀取、處理、寫入幀以及添加音訊。
        """
        # 設置寫入器以保存處理後的影片
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (self.crop_width, self.crop_height))

        # 遍歷原始影片的每一幀
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            processed_frame = self.process_frame(frame) # 處理幀
            out.write(processed_frame) # 寫入處理後的幀
        # 釋放資源
        self.cap.release()
        out.release()
        # 添加音訊到處理後的影片
        self.audio_manager.add_audio_to_video()
    
def main():
    """
    主函數，創建 VideoProcessor 實例並運行影片處理。
    """
    input_video_path = 'test.mp4'
    output_video_path = 'test_output_automation_per5.mp4'

    processor = VideoProcessor(input_video_path, output_video_path)
    processor.run() # 運行影片處理

if __name__ == '__main__':
    main()

    