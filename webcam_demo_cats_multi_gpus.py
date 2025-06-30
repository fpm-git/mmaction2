import time
from collections import deque
from operator import itemgetter
from threading import Thread, Event
from pathlib import Path
import cv2
import numpy as np
import torch
from mmengine import Config
from mmengine.dataset import Compose, pseudo_collate
from mmaction.apis import init_recognizer
from mmaction.utils import get_str_type
from mmaction.utils.stream_source import VideoStream
from websocket_server import WebsocketServer

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 2
FONTCOLOR = (0, 0, 255)  # BGR, RED
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 2
LINETYPE = 1
EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]

# global
stop_signal = Event()  # signal to terminate threads
active_streams_dict = {}
# num of frames used for moving average (affects smoothing of predictions)
average_size = 1
threshold = 0.01  # minimum confidence score for predictions to be displayed
shoot_threshold = 0.9
websocket_clients = []
websocket_stop_signal = Event()


def show_results(frame_queues, result_queues):
    """Display or save annotated frames"""
    print("[Info] Starting display for all videos")
    text_info_dict = {name: {} for name in frame_queues}

    while not stop_signal.is_set():
        msg = 'Waiting for action ...'
        for stream_name, frame_queue in frame_queues.items():
            if not frame_queue:
                continue

            frame = frame_queue[-1]  # get the latest frame from queue

            if len(result_queues[stream_name]) != 0:
                text_info_dict[stream_name] = {}
                results_data = result_queues[stream_name].popleft()
                results = results_data["scores"]

                for i, result in enumerate(results):
                    selected_label, score = result

                    # THIS IS WHERE THE SCRATCHING DECISION IS MADE FROM THE RESULTS
                    if score < threshold:
                        break

                    # TODO --> THis is where we need to inject a websocket server to send a shoot command to the turret
                    # cat turret does socket_queue.get(block=False) to get a message that says "Shoot" or no

                    location = (10, 40 + i * 30)
                    text = f"{selected_label}: {round(score * 100, 2)}"
                    text_info_dict[stream_name][location] = text
                    if score > shoot_threshold and selected_label == 'scratching':
                        print(f"[Info] Scratching detected for {stream_name}")

                        for client in websocket_clients:
                            try:
                                client.send("Shoot")
                            except Exception as e:
                                print(f"failed to send message {e}")

                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)

            elif len(text_info_dict[stream_name]) != 0:
                for location, text in text_info_dict[stream_name].items():
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)
            else:
                cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                            THICKNESS, LINETYPE)

            """save output images if using micron system"""
            # file_name = f'output_images/{stream_name}.jpg' #png
            # temp_file = f'output_images/{stream_name}_temp.jpg'

            # success = cv2.imwrite(temp_file, frame)
            # if success:
            #     os.replace(temp_file, file_name)

            # cv2.imwrite(file_name, frame)
            # frame_num += 1
            cv2.imshow(f"Camera - {stream_name}", frame)

        key = cv2.waitKey(1)
        if key == 27 or key in [ord('q'), ord('Q')]:
            print("[Info] Stopping all video streams...")
            stop_signal.set()
            break
    cv2.destroyAllWindows()


def inference(stream_name, frame_queue, result_queue, model, data, label, test_pipeline, sample_length):
    """Run inference on video frames and put results into result_queue"""
    score_cache = deque()
    scores_sum = 0
    while not stop_signal.is_set():
        cur_windows = []
        while len(cur_windows) == 0 and not stop_signal.is_set():
            if len(frame_queue) == sample_length:
                cur_windows = list(np.array(frame_queue))
                if data['img_shape'] is None:
                    data['img_shape'] = frame_queue.popleft().shape[:2]
            elif active_streams_dict[stream_name].stopped:
                return
        if stop_signal.is_set():
            break

        cur_data = data.copy()
        cur_data['imgs'] = cur_windows
        cur_data = test_pipeline(cur_data)
        cur_data = pseudo_collate([cur_data])

        try:
            with torch.no_grad():
                result = model.test_step(cur_data)[0]
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("[Error] CUDA runtime error")
                stop_signal.set()
                return
            else:
                raise

        scores = np.array(result.pred_score.tolist())
        score_cache.append(scores)
        scores_sum += scores

        if len(score_cache) == average_size:
            scores_avg = scores_sum / average_size
            num_selected_labels = min(len(label), 5)
            score_tuples = tuple(zip(label, scores_avg))
            score_sorted = sorted(
                score_tuples, key=itemgetter(1), reverse=True)
            results = score_sorted[:num_selected_labels]

            results_dict = {"frame_src": stream_name, "scores": results}
            result_queue.append(results_dict)
            scores_sum -= score_cache.popleft()


def initialize_model(config_path, checkpoint_path, device_id, label_path):
    """Initialize the action recognition model and preprocessing pipeline"""
    cfg = Config.fromfile(config_path)
    device = f'cuda:{device_id}'
    model = init_recognizer(cfg, checkpoint_path, device=device)

    sample_length = 0
    pipeline = cfg.test_pipeline.copy()
    data = dict(img_shape=None, modality='RGB', label=-1)

    for step in cfg.test_pipeline:
        if 'SampleFrames' in get_str_type(step['type']):
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline.remove(step)
        if get_str_type(step['type']) in EXCLUED_STEPS:
            pipeline.remove(step)

    test_pipeline = Compose(pipeline)
    assert sample_length > 0
    with open(label_path, 'r') as f:
        label = [line.strip() for line in f]
    return model, test_pipeline, label, sample_length, data


def fetch_frames(stream, frame_queue):
    """Read frames from a video file and put them into a frame queue"""
    while not stream.stopped and not stop_signal.is_set():
        frame, _, stopped = stream.get_frame()
        if stopped or stop_signal.is_set():
            break
        if frame is not None:
            frame_queue.append(frame)
        time.sleep(0.01)


def new_websocket_client(client, server):
    print("New client connected.")
    websocket_clients.append(client)
    while True:
        message = client.recv()
        if not message:
            break
        print(f"Received message: {message}")
        client.send(message)  # echo message back
    print("Client disconnected.")
    websocket_clients.remove(client)


def run_websocket_server():
    port = 8765
    server = WebsocketServer(host='localhost', port=port)
    server.set_fn_new_client(new_websocket_client)
    print(f"Websocket server started, listening on {port}")
    while not websocket_stop_signal.is_set():
        server.handle_request()  # handle requests
        time.sleep(0.1)  # Avoid busy waiting


def main():
    gpu_count = torch.cuda.device_count()
    print(f"[Info] Available GPUs: {gpu_count}")

    config_path = 'configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_cats_cropped.py'
    checkpoint_path = 'best_models/best_acc_top1_epoch_19_3_7.pth'
    label_path = 'cats_labels.txt'

    stream_sources = {
        "Video_1": 'test_videos/download_test_1.mp4',
        # "Video_2": 0,
    }

    frame_queues = {}
    result_queues = {}
    threads = []

    # create websocket for turret communication
    server_thread = Thread(target=run_websocket_server)
    threads.append(server_thread)

    for idx, (name, video_path) in enumerate(stream_sources.items()):
        gpu_id = idx % gpu_count  # assign GPUs
        gpu_name = torch.cuda.get_device_name(gpu_id)

        print(f"[Info] Loading model for {name} on GPU {gpu_id}: {gpu_name}")
        model, test_pipeline, label, sample_length, data = initialize_model(
            config_path, checkpoint_path, gpu_id, label_path
        )
        video_stream_widget = VideoStream(src=video_path)
        video_stream_widget.start()
        active_streams_dict[name] = video_stream_widget

        frame_queues[name] = deque(maxlen=sample_length)
        result_queues[name] = deque(maxlen=1)

        t_fetch = Thread(target=fetch_frames, args=(
            video_stream_widget, frame_queues[name]), daemon=True)
        t_infer = Thread(target=inference,
                         args=(name, frame_queues[name], result_queues[name],
                               model, data, label, test_pipeline, sample_length),
                         daemon=True)

        threads.extend([t_fetch, t_infer])

    t_show = Thread(target=show_results, args=(frame_queues, result_queues))
    threads.append(t_show)

    for t in threads:
        t.start()

    while any(not stream.stopped for stream in active_streams_dict.values()):
        time.sleep(1)
    stop_signal.set()
    websocket_stop_signal.set() # Signal to stop WebSocket server

    for t in threads:
        t.join()

    print("[Info] All threads stopped, exiting program.")


if __name__ == '__main__':
    main()
