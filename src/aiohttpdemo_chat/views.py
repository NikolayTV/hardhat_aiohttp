import base64
import io
import logging
import json
import aiohttp
import aiohttp_jinja2
import cv2
import numpy as np
from PIL import Image
from aiohttp import web
from faker import Faker
from core.HardHat_detection_yolov5 import Hardhat_detection_yolov5
import sys
sys.path.insert(0, '../core/')


log = logging.getLogger(__name__)

hardhat_detector = Hardhat_detection_yolov5(view=False, draw=True)

# Take in base64 string and return cv image
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


def get_random_name():
    fake = Faker()
    return fake.name()


async def user_video_capture(request):
    ws_current = web.WebSocketResponse()
    ws_ready = ws_current.can_prepare(request)
    if not ws_ready.ok:
        return aiohttp.WebSocketError()

    await ws_current.prepare(request)
    await ws_current.send_json({'action': 'connect'})

    while True:
        video_msg = await ws_current.receive()
        if video_msg.type == aiohttp.WSMsgType.BINARY:
            cap = cv2.VideoCapture(video_msg)
            while cap.isOpened():
                ret, bgr_image = cap.read()
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                frame_result, landmarks, headPosition, output, gaze_vector = driver_monitor.run_per_image(rgb_image)
                ret, jpeg = cv2.imencode('.jpg', frame_result)
                await ws_current.send_bytes(data=jpeg.tobytes())
            await ws_current.close()
        else:
           break

    return ws_current


async def rtsp_detection_stream(request):
    ws_current = web.WebSocketResponse()
    ws_ready = ws_current.can_prepare(request)
    if not ws_ready.ok:
        return aiohttp_jinja2.render_template('rtsp_template.html', request, {})

    await ws_current.prepare(request)
    await ws_current.send_json({'action': 'connect'})

    # vcap = cv2.VideoCapture("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")
    vcap = cv2.VideoCapture("../../data/Safety_Full_Hat_and_Vest.mp4")
    while (True):
        ret, frame = vcap.read()
        bboxes_xyx2y2, labels, frame_processed = hardhat_detector.predict(frame)
        ret, jpeg = cv2.imencode('.jpg', frame_processed)
        # Send back image in bytes
        await ws_current.send_bytes(data=jpeg.tobytes())

    # at closing delete from WS list
    log.info('user disconnected.')
    # Return
    return ws_current


async def index(request):
    # define connection
    ws_current = web.WebSocketResponse()
    ws_ready = ws_current.can_prepare(request)
    if not ws_ready.ok:
        return aiohttp_jinja2.render_template('index.html', request, {})

    # # For preparation
    await ws_current.prepare(request)

    # Random names
    name = get_random_name()
    log.info('%s joined.', name)

    # Send via current websocket success connection
    await ws_current.send_json({'action': 'connect', 'name': name})

    # For each websocket send message that new user connect
    for ws in request.app['websockets'].values():
        await ws.send_json({'action': 'join', 'name': name})

    # Add current websocket into list of all websockets
    request.app['websockets'][name] = ws_current

    # Main loop
    while True:
        log.info('pre-receive')
        # Wait for get some info from websocket
        msg = await ws_current.receive()
        log.info(msg.type)
        # Define masg is BINARY
        if msg.type == aiohttp.WSMsgType.BINARY:
            # BINARY is an image
            # Inference with preparration
            image = Image.open(io.BytesIO(msg.data))
            opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
            frame_result, landmarks, headPosition, output, gaze_vector = driver_monitor.run_per_image(opencvImage)
            ret, jpeg = cv2.imencode('.jpg', frame_result)
            print(headPosition)
            # Send back image in bytes
            await ws_current.send_bytes(data=jpeg.tobytes())
            await ws_current.send_json({'action': 'cv_result', 'landmarks': 'landmark'})
        else:
            break

    # at closing delete ffrom WS list
    del request.app['websockets'][name]
    log.info('%s disconnected.', name)
    # Send to all users that websocket closed
    for ws in request.app['websockets'].values():
        await ws.send_json({'action': 'disconnect', 'name': name})
    # Return
    return ws_current


# Video uploading
# How to do this shit
# Upload video via websocket for current client
# Get Video stream
# Just loaded image
