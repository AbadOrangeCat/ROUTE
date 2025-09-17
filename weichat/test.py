# pip install pywin32 pyttsx3
import time

print("=== 方案A：Win32 SAPI.SpVoice 测试 ===")
try:
    import win32com.client  # from pywin32
    spk = win32com.client.Dispatch("SAPI.SpVoice")
    # 尝试选择中文语音
    for v in spk.GetVoices():
        vid = v.GetAttribute("Name") + " | " + v.GetAttribute("Language")
        print("  voice:", vid)
    # 优先中文
    for v in spk.GetVoices():
        name = v.GetAttribute("Name").lower()
        lang = v.GetAttribute("Language").lower()
        if "zh" in name or "804" in lang or "zh" in lang:
            spk.Voice = v
            break
    spk.Rate = 0
    spk.Volume = 100
    print("[SAPI] 说：T T S 正常")
    spk.Speak("T T S 正常")
    time.sleep(0.5)
    print("[SAPI] 说：你好，语音测试")
    spk.Speak("你好，语音测试")
except Exception as e:
    print("[SAPI] 失败：", e)

print("\n=== 方案B：pyttsx3 测试 ===")
try:
    import pyttsx3
    eng = pyttsx3.init()
    print("  可用语音：")
    for v in eng.getProperty('voices'):
        print("   -", v.id, getattr(v, "name", ""))
    # 选中文
    zh_id = None
    for v in eng.getProperty('voices'):
        name = (getattr(v, 'name', '') or '').lower()
        langs = ''.join(getattr(v, 'languages', [])).lower() if hasattr(v, 'languages') else ''
        vid = (getattr(v, 'id', '') or '').lower()
        if 'zh' in name or 'chinese' in name or 'zh' in langs or 'zh' in vid:
            zh_id = v.id
            break
    if zh_id:
        eng.setProperty('voice', zh_id)
    eng.setProperty('rate', 185)
    eng.setProperty('volume', 1.0)
    print("[pyttsx3] 说：你好，语音测试")
    eng.say("你好，语音测试")
    eng.runAndWait()
except Exception as e:
    print("[pyttsx3] 失败：", e)
