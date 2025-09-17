import pyttsx3

# 初始化语音引擎
engine = pyttsx3.init()

# 设置语音参数
engine.setProperty("rate", 160)   # 语速
engine.setProperty("volume", 1.0) # 音量

# 输入文字转语音
text = "你好，我是通过文字转语音和你交流的"
engine.say(text)
engine.runAndWait()
