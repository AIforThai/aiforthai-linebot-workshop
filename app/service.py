from fastapi import APIRouter, Request
from . import infer
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextSendMessage, ImageMessage
import PIL
router = APIRouter(tags=[""])


LINE_CHANNEL_ACCESS_TOKEN = ""
LINE_CHANNEL_SECRET = ""


line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)  # CHANNEL_ACCESS_TOKEN
handler = WebhookHandler(LINE_CHANNEL_SECRET)  # CHANNEL_SECRET


session, mode, config = infer.get_onnx_session('model-onnx')


@router.post("/message")
async def hello_word(request: Request):
    signature = request.headers["X-Line-Signature"]
    body = await request.body()
    try:
        handler.handle(body.decode("UTF-8"), signature)
    except InvalidSignatureError:
        print(
            "Invalid signature. Please check your channel access token or channel secret."
        )
    return "OK"


@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_id = event.message.id
    image_content = line_bot_api.get_message_content(message_id)

    # Save the image locally and process it
    with open("image.jpg", "wb") as f:
        for chunk in image_content.iter_content():
            f.write(chunk)

    # NomadML Inference section
    with PIL.Image.open("image.jpg") as _img:
        _img = infer.resize_keep_ratio(_img)
        res = infer.predict_image_onnx(session, mode, config, _img, 0.8)
    result = res

    # return text response
    send_message(event, result)


def echo(event):
    line_bot_api.reply_message(
        event.reply_token, TextSendMessage(text=event.message.text)
    )


# function for sending message
def send_message(event, message):
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=message))
