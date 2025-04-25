from fastapi import APIRouter, Request

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage, AudioSendMessage,AudioMessage

from aift import setting
from aift.multimodal import textqa

from datetime import datetime

from app.configs import Configs

from datetime import datetime

# AIForThai import
from aift.nlp import tokenizer # 1. Tokenizer
from aift.nlp import ner # 1.1 TNER
from aift.nlp import g2p # 2. G2P
from aift.nlp import soundex # 3. Soundex
from aift.nlp import similarity # 4. Word similarity
from aift.nlp import text_cleansing # 5. Text cleasing
from aift.nlp import tag # 6. Tag Suggestion
from aift.nlp.translation import zh2th # 7.1. Chinese to Thai
from aift.nlp.translation import th2zh # 7.2. Thai to Chinese
from aift.nlp.translation import en2th # 7.3. English to Thai
from aift.nlp.translation import th2en # 7.4. Thai to English
from aift.nlp import sentiment # 8. Sentiment analysis
from aift.nlp.longan import sentence_tokenizer, tagger, token_tagger, tokenizer as logan_tokenizer # 9. Longan
from aift.nlp.alignment import en_alignment # 10.1. English-Thai Word Aligner
from aift.nlp.alignment import zh_alignment # 10.2. Chinese-Thai Word Aligner
from aift.speech import tts

# For Partii STT
import io
import re
import json
import requests

# For Vaja9
import wave

router = APIRouter(tags=["NLP"], prefix="/nlp")

cfg = Configs()

setting.set_api_key(cfg.AIFORTHAI_APIKEY) # AIFORTHAI_APIKEY
line_bot_api = LineBotApi(cfg.LINE_CHANNEL_ACCESS_TOKEN)  # CHANNEL_ACCESS_TOKEN
handler = WebhookHandler(cfg.LINE_CHANNEL_SECRET)  # CHANNEL_SECRET


@router.post("")
async def message_qa(request: Request):
    signature = request.headers["X-Line-Signature"]
    body = await request.body()
    try:
        handler.handle(body.decode("UTF-8"), signature)
    except InvalidSignatureError:
        print(
            "Invalid signature. Please check your channel access token or channel secret."
        )
    return "OK"

@handler.add(MessageEvent, message=AudioMessage)
def handle_voice_message(event):
    # #14. SPEECH TO TEXT (Partii)
    # Get the audio file from LINE
    message_content = line_bot_api.get_message_content(event.message.id)
    
    # Save the audio content to a file (optional, for further processing)
    with open("received_audio.wav", "wb") as f:
        for chunk in message_content.iter_content():
            f.write(chunk)
        
    text = callPartii("received_audio.wav")
    send_message(event,str(text))


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_input = event.message.text.strip()
    command_list = [
        "#trexplus","#lexto", "#trex++", "#tner","#g2p", "#soundex","#thaiwordsim", "#wordapprox", 
        "#textclean", "#tagsuggest", "#mtch2th", "#mtth2ch", "#mten2th", "#mtth2en", "#ssense", "#emonews",
        "#thaimoji", "#cyberbully", "#longan_sentence", "#longan_tagger", "#longan_tokentag", "#longan_tokenizer",
        "#en2th_aligner", "#ch2th_aligner","#tts"
    ]

    matched_command = None
    for cmd in command_list:
        if user_input.startswith(cmd):
            matched_command = cmd
            break

    if matched_command:
        content = user_input[len(matched_command):].strip()

        if matched_command == "#trexplus":
            result = tokenizer.tokenize(content, engine='trexplus', return_json=True)
            send_message(event, str(result))

        elif matched_command == "#lexto":
            result = tokenizer.tokenize(content, engine='trexplus', return_json=True)
            send_message(event, str(result))
        
        elif matched_command == "#trex++":
            result = tokenizer.tokenize(content, engine='trexplusplus', return_json=True)
            send_message(event, str(list(zip(result['words'], result['tags']))))
        
        elif matched_command == "#tner":
            result = ner.analyze(content, return_json=True)
            send_message(event, str(list(zip(result['words'], result['POS'], result['tags']))))
        
        elif matched_command == "#g2p":
            result = g2p.analyze(content)['output']['result']
            send_message(event, str(result))
        
        elif matched_command == "#soundex":
            matched = re.match(r"#soundex(?:_([a-zA-Z0-9]+))?(.*)", user_input)
            if matched:
                model = matched.group(1) if matched.group(1) else "personname"  # Default model, model = personname, royin
                content = matched.group(2).strip()

                result = soundex.analyze(content, model=model)['words']
                send_message(event, str(result))

        elif matched_command.startswith("#thaiwordsim"):
            matched = re.match(r"#thaiwordsim(?:_([a-zA-Z0-9]+))?(.*)", user_input)
            if matched:
                model = matched.group(1) if matched.group(1) else "thwiki"
                content = matched.group(2).strip()
                result = similarity.similarity(content, engine='thaiwordsim', model=model)
                send_message(event, str(result))         

        elif matched_command == "#wordapprox":
            matched = re.match(r"#wordapprox(?:_([a-zA-Z0-9]+))?(.*)", user_input)
            if matched:
                model = matched.group(1) if matched.group(1) else "personname"  # default to personname,  model = personname, royin, food
                content = matched.group(2).strip()

                result = similarity.similarity(content, engine='wordapprox', model=model, return_json=True)
                send_message(event, str(result))

        elif matched_command == "#textclean":
            result = text_cleansing.clean(content)
            send_message(event, str(result))
        
        elif matched_command == "#tagsuggest":
            result = tag.analyze(content, numtag=5)
            send_message(event, str(result))
        
        elif matched_command == "#mtch2th":
            # result = zh2th.translate(content, return_json=True)
            result = Chainess2Thai(content, "zh", "th")
            send_message(event, str(result))

        elif matched_command == "#mtth2ch":
            # result = th2zh.translate(content, return_json=True)
            result = Chainess2Thai(content, "th", "zh")
            send_message(event, str(result))

        elif matched_command == "#mten2th":
            # result = en2th.translate(content)
            result = translate_xiaofan(content, "en2th")
            send_message(event, str(result))

        elif matched_command == "#mtth2en":
            # result = th2en.translate(content)
            result = translate_xiaofan(content, "th2en")
            send_message(event, str(result))

        elif matched_command == "#ssense":
            result = sentiment.analyze(content, engine='ssense')
            send_message(event, str(result))
        
        elif matched_command == "#emonews":
            result = sentiment.analyze(content, engine='emonews')
            send_message(event, str(result))
        
        elif matched_command == "#thaimoji":
            result = sentiment.analyze(content, engine='thaimoji')
            send_message(event, str(result))
        
        elif matched_command == "#cyberbully":
            result = sentiment.analyze(content, engine='cyberbully')
            send_message(event, str(result))
        
        elif matched_command == "#longan_sentence":
            result = sentence_tokenizer.tokenize(content)
            send_message(event, str(result))
        
        elif matched_command == "#longan_tagger":
            result = tagger.tag(content)
            send_message(event, str(result))
        
        elif matched_command == "#longan_tokentag":
            result = token_tagger.tokenize_tag(content)
            send_message(event, str(result))
        
        elif matched_command == "#longan_tokenizer":
            result = logan_tokenizer.tokenize(content)
            send_message(event, str(result))
        
        elif matched_command == "#en2th_aligner":
            # # ตัวอย่างภาษาอังกฤษ-ไทย เช่น "I like to recommend my friends to Thai restaurants|ฉันชอบแนะนำเพื่อนไปร้านอาหารไทย"
            contents = content.split('|') # รับข้อความจาก Line ในรูปแบบคู่ภาษาที่ต้องการจับคู่ ด้วยเครื่องหมาย "|"
            result = en_alignment.analyze(contents[0], contents[1], return_json=True)
            send_message(event, str(result))

        elif matched_command == "#ch2th_aligner":
            # # ตัวอย่างภาษาจีน-ไทย เช่น "我是10月10日从泰国来的。|ฉันมาจากประเทศไทยเมื่อวันที่ 10 เดือนตุลาคม"
            contents = content.split('|') # รับข้อความจาก Line ในรูปแบบคู่ภาษาที่ต้องการจับคู่ ด้วยเครื่องหมาย "|"
            result = zh_alignment.analyze(contents[0], contents[1], return_json=True)
            send_message(event, str(result))
        
        
        elif matched_command == "#tts":
            speaker = 0
            response = callVaja9(content, speaker)
            if response.json()['msg'] == 'success':
                download_and_play(response.json()['wav_url'])
                audio_url = cfg.WAV_URL + cfg.DIR_FILE + cfg.WAV_FILE
                duration_ms = int(response.json()['durations'] * 1000)
                audio_message = AudioSendMessage(
                    original_content_url=audio_url,
                    duration=duration_ms
                )
                send_audio_message(event, audio_message)
            else:
                send_message(event, "TTS failed")
    else:
        # echo(event)
        send_message(event, "Service not found")
     
def echo(event):
    line_bot_api.reply_message(
        event.reply_token, TextSendMessage(text=event.message.text)
    )


# function for sending audio message
def send_audio_message(event,audio_message):
        line_bot_api.reply_message(
            event.reply_token,
            audio_message)
        
# function for sending message
def send_message(event, message):
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=message))

def get_wav_duration_in_ms(file_path):
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = int(frames / rate) * 1000  # Convert seconds to milliseconds
        return duration

# Function call Vaja9
def callVaja9(text, speaker):
    url = cfg.URL_VAJA

    headers = {'Apikey':cfg.AIFORTHAI_APIKEY,"Content-Type": "application/json"}
    data = {'input_text':text,'speaker': speaker}
    response = requests.post(url, json=data, headers=headers)
    return response

# Function for download audio file
def download_and_play(sWav_url):
    file_name = cfg.DIR_FILE+cfg.WAV_FILE
    with open(file_name, 'wb') as a:
        resp = requests.get(sWav_url,headers={'Apikey':cfg.AIFORTHAI_APIKEY})
        # print(resp.status_code)
        if resp.status_code == 200:
            a.write(resp.content)
            #   print('Downloaded: '+sWav_url)
        else:
            print(resp.reason)
            exit(1)
    # return file_name
    
def get_wav_duration_in_ms(file_path):
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = (frames / rate) * 1000  # Convert seconds to milliseconds
        return duration

# Function for call Partii
def callPartii(file):
    url = cfg.URL_PARTII

    files = {'wavfile': (file, open(file, 'rb'), 'audio/wav')}

    headers = {
            'Apikey': cfg.AIFORTHAI_APIKEY,
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            }

    param = {"outputlevel":"--uttlevel","outputformat":"--txt"}

    response = requests.request("POST", url, headers=headers, files=files, data=param)
    data = json.loads(response.text)
    return data['message']

# Function call Chinese to Thai/ Thai to Chinese
def Chainess2Thai(text, src, tar):
    url = "https://api.aiforthai.in.th/xiaofan-zh-th"
 
    payload = json.dumps({
    "input": text,
    "src": src,
    "trg": tar
    })
    headers = {'apikey': cfg.AIFORTHAI_APIKEY,'Content-Type': 'application/json'}
    
    response = requests.request("POST", url, headers=headers, data=payload)
    # print(response.json())
    return response.json()['output']

# Function call English to Thai/ Thai to English
def translate_xiaofan(text, direction):
    # direction = 'en2th' or 'th2en'
    url = f"https://api.aiforthai.in.th/xiaofan-en-th/{direction}"

    # Payload key changes based on direction
    if direction in ["en2th", "th2en"]:
        payload = json.dumps({"text": text})
    else:
        raise ValueError("Invalid direction. Use 'en2th' or 'th2en'.")

    headers = {
        'apikey': cfg.AIFORTHAI_APIKEY,
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=payload)
    return response.json()['translated_text']  # or response.text if you prefer raw

# Function call Text summarization
def callTextSummarization(content):
    url = 'http://api.aiforthai.in.th/textsummarize'
    headers = {'Apikey': cfg.AIFORTHAI_APIKEY, 'Content-Type': 'application/json'}
    params = json.dumps([{"id": 100, "comp_rate": 30, "src": content}])
    response = requests.post(url, data=params, headers=headers)
    txt = response.text
    text_sum = bytes(txt, "utf-8").decode("unicode_escape")
    return text_sum


    
# End of  file
