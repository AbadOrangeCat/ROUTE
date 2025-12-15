from pydub import AudioSegment
from pydub.utils import which
import whisper
import os

def main():
    # 1. 指定 mp3 路径
    mp3_path = r"audio1060454005.mp3"  # 改成你的 mp3 文件路径
    wav_path = os.path.splitext(mp3_path)[0] + ".wav"      # 自动生成 wav 路径

    # 2. 配置 ffmpeg 路径（如果已在 PATH，可省略）
    AudioSegment.converter = which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
    AudioSegment.ffprobe   = which("ffprobe") or r"C:\ffmpeg\bin\ffprobe.exe"

    # 3. mp3 → wav
    print("正在将 mp3 转换为 wav...")
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")
    print(f"转换完成: {wav_path}")

    # 4. 加载 Whisper 模型
    print("正在加载 Whisper 模型，请稍等...")
    model = whisper.load_model("small")  # 可选 tiny / base / small / medium / large

    # 5. 翻译（英文 → 中文）
    print("正在识别并翻译音频...")
    result = model.transcribe(wav_path, task="translate")

    # 6. 打印结果
    print("\n======= 中文翻译结果 =======\n")
    print(result["text"])
    print("\n===========================")

if __name__ == "__main__":
    main()
