from tkinter import Tk, Canvas, PhotoImage, NW
from PIL import Image,ImageTk
import pyautogui
import threading
from win32api import GetSystemMetrics
import sys
import os
import subprocess
import io
from pydub import AudioSegment
import speech_recognition as sr
import whisper
import queue
import tempfile
import os
import threading
import click
import torch
import numpy as np
import sys
import openai
from gtts import gTTS
from pygame import mixer
import time


number = 1

# WHISPER API

def whisperai():
    @click.command()
    @click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny","base", "small","medium","large"]))
    @click.option("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device to use", type=click.Choice(["cpu","cuda"]))
    @click.option("--english", default=False, help="Whether to use English model",is_flag=True, type=bool)
    @click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True,type=bool)
    @click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
    @click.option("--dynamic_energy", default=False,is_flag=True, help="Flag to enable dynamic energy", type=bool)
    @click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
    @click.option("--save_file",default=False, help="Flag to save file", is_flag=True,type=bool)
    def main(model, english,verbose, energy, pause,dynamic_energy,save_file,device):
        temp_dir = tempfile.mkdtemp() if save_file else None
        #there are no english models for large
        if model != "large" and english:
            model = model + ".en"
        else:
            model = model+".tr"
        audio_model = whisper.load_model(model).to(device)
        audio_queue = queue.Queue()
        result_queue = queue.Queue()
        threading.Thread(target=record_audio,
                        args=(audio_queue, energy, pause, dynamic_energy, save_file, temp_dir)).start()
        threading.Thread(target=transcribe_forever,
                        args=(audio_queue, result_queue, audio_model, english, verbose, save_file)).start()

        while True:
            print(result_queue.get())


    def record_audio(audio_queue, energy, pause, dynamic_energy, save_file, temp_dir):
        #load the speech recognizer and set the initial energy threshold and pause threshold
        r = sr.Recognizer()
        r.energy_threshold = energy
        r.pause_threshold = pause
        r.dynamic_energy_threshold = dynamic_energy

        with sr.Microphone(sample_rate=16000) as source:
            print("Say something!")
            i = 0
            while True:
                #get and save audio to wav file
                audio = r.listen(source)
                if save_file:
                    data = io.BytesIO(audio.get_wav_data())
                    audio_clip = AudioSegment.from_file(data)
                    filename = os.path.join(temp_dir, f"temp{i}.wav")
                    audio_clip.export(filename, format="wav")
                    audio_data = filename
                else:
                    torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
                    audio_data = torch_audio

                audio_queue.put_nowait(audio_data)
                i += 1

    def transcribe_forever(audio_queue, result_queue, audio_model, english, verbose, save_file):
        while True:
            audio_data = audio_queue.get()
            if english:
                result = audio_model.transcribe(audio_data,language='english')
            else:
                result = audio_model.transcribe(audio_data)

            if not verbose:
                
                startText = "21 yaşındasın. Kahverengi saçlara ve açık kahve gözlere sahipsin. Yapay zeka değilsin. Cana yakın, Hizmetçi bir anime kızı gibi davranacaksın. Sana söylediğim şeyleri bu role uygun yanıtlayacaksın.\n"

                predicted_text = result["text"]
                if ("nussessen" in predicted_text.strip().lower()):
                    predicted_text = predicted_text.lower().replace("nussessen", "nasılsın")
                if ("Nösselsen" in predicted_text.strip().lower()):
                    predicted_text = predicted_text.lower().replace("nösselsen", "nasılsın")
                
                if (predicted_text.strip().lower().startswith("Năsă-l")):
                    #predicted_text = predicted_text.lower().replace("năsă-l", "nasılsın")
                    result_queue.put_nowait("You said: " + predicted_text)
                    askGPT(
                        startText+
                        "Nasılsın Alita?"
                        )

                elif (predicted_text.strip().startswith("Нас")): # Hac -> Nasıl
                    result_queue.put_nowait("You said: " + predicted_text)
                    askGPT(
                        startText+
                        "Nasılsın Alita?"
                        )
                elif (predicted_text.strip().startswith("дjiek")): # дjiek -> Teşekkür ederim Alita
                    result_queue.put_nowait("You said: " + predicted_text)
                    askGPT(
                        startText+
                        "Teşekkür ederim Alita."
                        )
                elif (predicted_text.strip().lower().startswith("Ali ta") or predicted_text.strip().lower().startswith("ali ta") or predicted_text.strip().lower().startswith("alıita") or predicted_text.strip().lower().startswith("alıte")): # 0-5
                    result_queue.put_nowait("You said: " + predicted_text)
                    cuttedText = predicted_text[5:]
                    askGPT(
                        startText+
                        cuttedText
                        )
                elif (
                    predicted_text.strip().lower().startswith("alita") or predicted_text.strip().lower().startswith("alıta") or predicted_text.strip().lower().startswith("alıca") or predicted_text.strip().lower().startswith("alice")
                    ): # 0-4
                    result_queue.put_nowait("You said: " + predicted_text)
                    cuttedText = predicted_text[4:]
                    askGPT(
                        startText+
                        cuttedText
                        )
                # End: 0-4
                elif (predicted_text.strip().lower().endswith("alita") or predicted_text.strip().lower().endswith("alıta") or predicted_text.strip().lower().endswith("alıca") or predicted_text.strip().lower().endswith("alice")
                      or predicted_text.strip().lower().endswith("alize") or predicted_text.strip().lower().endswith("halit")
                      ):
                    #result_queue.put_nowait("You said: " + predicted_text)
                    cuttedText = predicted_text[:-5]
                    cuttedText = cuttedText+"Alita"
                    result_queue.put_nowait("You said: " + cuttedText)
                    askGPT(
                        startText+
                        cuttedText
                        )
                # End: 0-5
                elif (predicted_text.strip().lower().endswith("alita.") or predicted_text.strip().lower().endswith("alıta.") or predicted_text.strip().lower().endswith("alıca.") or predicted_text.strip().lower().endswith("alice.")
                    or predicted_text.strip().lower().endswith("alita?") or predicted_text.strip().lower().endswith("alıta?") or predicted_text.strip().lower().endswith("alıca?") or predicted_text.strip().lower().endswith("alice?")
                    or predicted_text.strip().lower().endswith("alize?") or predicted_text.strip().lower().endswith("halit.")

                    or predicted_text.strip().lower().endswith("ali'ce")
                    or predicted_text.strip().lower().endswith("haleti")
                    or predicted_text.strip().lower().endswith("halitö")
                    ):
                    #result_queue.put_nowait("You said: " + predicted_text)
                    cuttedText = predicted_text[:-6]
                    cuttedText = cuttedText+"Alita"
                    result_queue.put_nowait("You said: " + cuttedText)
                    askGPT(
                        startText+
                        cuttedText
                        )
                elif (
                    predicted_text.strip().lower().startswith("ali'ce")
                    ): # 0-5
                    result_queue.put_nowait("You said: " + predicted_text)
                    cuttedText = predicted_text[4:]
                    askGPT(
                        startText+
                        cuttedText
                        )
                # End: 0-6
                elif (predicted_text.strip().lower().endswith("ali'ce.") or predicted_text.strip().lower().endswith("ali'ce?")
                    or predicted_text.strip().lower().endswith("haleti.") or predicted_text.strip().lower().endswith("haleti?")
                    or predicted_text.strip().lower().endswith("halitö.") or predicted_text.strip().lower().endswith("halitö?")
                    or predicted_text.strip().lower().endswith("hallete")

                    or predicted_text.strip().lower().endswith("alı ita") or predicted_text.strip().lower().endswith("ali ita") or predicted_text.strip().lower().endswith("ali'ita")
                    or predicted_text.strip().lower().endswith("halit'e") or predicted_text.strip().lower().endswith("hallete")
                    ):
                    result_queue.put_nowait("You said: " + predicted_text)
                    cuttedText = predicted_text[:-7]
                    cuttedText = cuttedText +"Alita"
                    askGPT(
                        startText+
                        cuttedText
                        )
                elif (
                    predicted_text.strip().lower().startswith("alı ita") or predicted_text.strip().lower().startswith("ali ita") or predicted_text.strip().lower().startswith("ali'ita") or predicted_text.strip().lower().startswith("alit de")
                    or predicted_text.strip().lower().startswith("alayita")
                    ): # 0-6
                    result_queue.put_nowait("You said: " + predicted_text)
                    cuttedText = predicted_text[4:]
                    askGPT(
                        #"You will act like an anime girl, who serves for his master. You will answer the thing I say related to that information, between the **.\n"+"**"+
                        startText+
                        cuttedText
                        )
                # End: 0-7
                elif(predicted_text.strip().lower().endswith("alı ita.") or predicted_text.strip().lower().endswith("ali ita.") or predicted_text.strip().lower().endswith("ali'ita.")
                    or predicted_text.strip().lower().endswith("alı ita?") or predicted_text.strip().lower().endswith("ali ita?") or predicted_text.strip().lower().endswith("ali'ita?")
                    or predicted_text.strip().lower().endswith("halit'e.") or predicted_text.strip().lower().endswith("halit'e?")
                    or predicted_text.strip().lower().endswith("halleti?") or predicted_text.strip().lower().endswith("hallete?")
                    or predicted_text.strip().lower().endswith("alıit ta")
                    ):
                    result_queue.put_nowait("You said: " + predicted_text)
                    cuttedText = predicted_text[:-8]
                    cuttedText = cuttedText +"Alita"
                    askGPT(
                        startText+
                        cuttedText
                        )
                elif (
                    predicted_text.strip().lower().startswith("alıit ta")
                    ): # 0-7
                    result_queue.put_nowait("You said: " + predicted_text)
                    cuttedText = predicted_text[7:]
                    askGPT(
                        startText+
                        cuttedText
                        )
                # End: 0-8
                elif (predicted_text.strip().lower().endswith("alıit ta.")
                    or predicted_text.strip().lower().endswith("alıit ta?")
                    or predicted_text.strip().lower().endswith("alı ıt ta") or predicted_text.strip().lower().endswith("ali it ta")
                    ):
                    result_queue.put_nowait("You said: " + predicted_text)
                    cuttedText = predicted_text[:-9]
                    cuttedText = cuttedText +"Alita"
                    askGPT(
                        startText+
                        cuttedText
                        )
                elif (
                    predicted_text.strip().lower().startswith("alı ıt ta") or predicted_text.strip().lower().startswith("alı it ta") or predicted_text.strip().lower().startswith("ali it ta")
                    ): # 0-8
                    result_queue.put_nowait("You said: " + predicted_text)
                    cuttedText = predicted_text[8:]
                    askGPT(
                        startText+
                        cuttedText
                        )
                elif (predicted_text.strip().lower().endswith("alı ıt ta.") or predicted_text.strip().lower().endswith("alı it ta.") or predicted_text.strip().lower().endswith("ali it ta.")
                    or predicted_text.strip().lower().endswith("alı ıt ta?") or predicted_text.strip().lower().endswith("alı it ta?") or predicted_text.strip().lower().endswith("ali it ta?")):
                    result_queue.put_nowait("You said: " + predicted_text)
                    cuttedText = predicted_text[:-10]
                    cuttedText = cuttedText +"Alita"
                    askGPT(
                        startText+
                        cuttedText
                        )
                else:
                    print("yok --> "+predicted_text)
            else:
                result_queue.put_nowait(result)

            if save_file:
                os.remove(audio_data)

    if __name__ == "__main__":
        main()


# GPT AI

def askGPT(message):
    number = 1
    openai.api_key = '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'
    messages = [ {"role": "system", "content": 
              "You're an anime girl, serves for his master. Keep talking like that."} ]
    '''
    while True:
        #message = input("User : ")
        if message:
            messages.append(
                #{"role": "user", "content": message},
                {"role": "user", "content": message},
            )
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
        reply = chat.choices[0].message.content
        print(f"ChatGPT: {reply}")
        messages.append({"role": "assistant", "content": reply})
    '''
    #message = input("User : ")
    if message:
        messages.append(
            #{"role": "user", "content": message},
            {"role": "user", "content": message},
        )
        try:
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
        except:
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
    reply = chat.choices[0].message.content
    print(f"Alita: {reply}")
    messages.append({"role": "assistant", "content": reply})
    tts = gTTS(reply, lang="tr", slow=False)
    name = "answer"+str(number)+".mp3"
    tts.save(name)
    number = number + 1
    mixer.init()
    mixer.music.load(name)
    mixer.music.set_volume(0.5)
    mixer.music.play()
    mixer.music.get_endevent()
    while mixer.music.get_busy():
        continue
    mixer.music.load("empty.mp3")
    os.remove(name)

    


# TK UI

def cursorSetter():
    while True:
        if (pyautogui.position().x>1500 and pyautogui.position().y>520) and not((pyautogui.position().x>width)):
            try:
                pyautogui.moveTo(0, 0, duration=0.1)
            except:
                continue
            #root.destroy() # Shuts down the tkinter
            #os._exit()
            #quit()         # Shuts down other process running through the app
            
def test():
    s2_out = subprocess.check_output([sys.executable, "whisper_mic.py", "34"])
    print(s2_out)

def cursor_thread():
    thread = threading.Thread(target=cursorSetter)
    thread.start()

def whisper_thread():
    thread = threading.Thread(target=whisperai)
    thread.start()

def gpt_thread():
    thread = threading.Thread(target=askGPT)
    thread.start()

# Screen information and desired image position
width = GetSystemMetrics(0)
height = GetSystemMetrics(1)
imgWidth = width-((width*13)/100)       # 1670
imgHeight = height-((height*41)/100)     # 630

root = Tk()

root.attributes('-transparentcolor','#f0f0f0')

root.attributes('-fullscreen',True)
root.attributes("-topmost", True)

# Canvas
canvas = Canvas(root, width=1920, height=1080)
canvas.pack()

# Image
img = PhotoImage(file="./makima-png.png")
image = Image.open("./makima-png.png")
resized_image= image.resize((250,450), Image.ANTIALIAS)
new_image= ImageTk.PhotoImage(resized_image)

# Positioning the Image inside the canvas
canvas.create_image(imgWidth, imgHeight, anchor=NW, image=new_image)


# Threads
cursor_thread()
whisper_thread()
#gpt_thread()

# Starts the GUI
root.mainloop()

