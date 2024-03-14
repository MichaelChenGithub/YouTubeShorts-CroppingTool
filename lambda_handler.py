import boto3
from video_processor import VideoProcessor

def lambda_handler(event, context):
    s3_client = boto3.client('s3')

    # 假设 event 中包含了 S3 bucket 名称和视频文件键名
    bucket_name = event['bucket']
    input_key = event['input_key']
    output_key = event['output_key']

    # 下载视频文件
    input_video_path = '/tmp/input_video.mp4'
    s3_client.download_file(bucket_name, input_key, input_video_path)

    # 处理视频
    output_video_path = '/tmp/output_video.mp4'
    processor = VideoProcessor(input_video_path, output_video_path)
    processor.run()

    # 上传处理后的视频
    s3_client.upload_file(output_video_path, bucket_name, output_key)

    return {
        'statusCode': 200,
        'body': 'Video processing completed successfully.'
    }
