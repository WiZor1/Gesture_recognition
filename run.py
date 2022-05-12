import os
from pathlib import Path
from IPython.display import clear_output

import cv2
from PIL import Image
from facenet_pytorch import MTCNN

import numpy as np

import torch
from torch.nn import functional as F
from torchvision import transforms
import imageio
import datetime as dt

from model_define import SignDataset, MyResNet, df_transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


# TODO: docstrings...

class GestureRecog():
    def __init__(self,
                 gesture_model: torch.nn.Module,
                 face_detect_model: torch.nn.Module = None,
                 transform: transforms.transforms.Compose = None,
                 chain_delay: int = 5,
                 face_detect_freq: int = 10,
                 sign_dict: dict = None,
                 sign_pics: dict = None,
                 pic_size: tuple = (320, 240),
                 text_color: tuple = (255, 255, 0),
                 action_delay: int = 3,
                 save_imgs: bool = True,
                 PATH=Path('.'),
                 RESULTS_PATH: Path = Path('results'),
                 ICONS_PATH: Path = Path('icons'),
                 device='cpu'
                 ):
        self._RESULTS_PATH = RESULTS_PATH
        self._ICONS_PATH = ICONS_PATH
        self._RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        self._ICONS_PATH.mkdir(parents=True, exist_ok=True)

        self._gesture_model = gesture_model
        if face_detect_model is None:
            self._face_detect_model = MTCNN(image_size=pic_size).to(device)
        else:
            self._face_detect_model = face_detect_model
        if transform is None:
            self._transform = df_transforms
        else:
            self._transform = transform

        self._res = None
        self._chain_delay = chain_delay
        self._face_detect_freq = face_detect_freq

        if sign_dict is None:
            self._sign_dict = {0: 'Nothing',
                               1: 'Minus',
                               2: 'Greetings',
                               3: 'Ok',
                               4: 'Thumb up',
                               5: 'Fist',
                               6: 'Index',
                               7: 'Two fingers'}
        else:
            self._sign_dict = sign_dict
        if sign_pics is None:
            self._sign_pics = {i: imageio.imread(self._ICONS_PATH / f'{self._sign_dict[i]}.png') for i in
                               self._sign_dict.keys() if i != 0}
        else:
            self._sign_pics = sign_pics

        self.actions = {1: (self._default_1_act, 'fixed', {'text': 'Cancelled', 'display_time': 10, 'act_time': 0}),
                        2: (self._default_2_act, 'fixed', {'text': 'Hello there!', 'display_time': 30, 'act_time': 0}),
                        3: (self._default_3_act, 'fixed',
                            {'text': 'Calc is opening...', 'display_time': 30, 'act_time': 30}),
                        4: (self._default_4_act, 'fixed',
                            {'text': 'Session is ending...', 'display_time': 30, 'act_time': 30}),
                        5: (self._default_5_act, 'continues',
                            {'text': 'Icon is decreasing...', 'display_time': 1, 'act_time': 0}),
                        6: (self._default_6_act, 'continues',
                            {'text': 'Icon is increasing...', 'display_time': 1, 'act_time': 0}),
                        7: (self._default_7_act, 'continues', {'text': 'Just test', 'display_time': 1, 'act_time': 0})}

        self._action_classification()

        self._pic_size = pic_size
        self._text_color = text_color
        self._action_delay = action_delay
        self._save_imgs = save_imgs

        self._resize_to = 100
        self._face_detected = False

    def _action_classification(self):
        self._cont_actions = [k for k, v in self.actions.items() if v[1] == 'continues']
        self._fixed_actions = [k for k, v in self.actions.items() if v[1] == 'fixed']

    def _query_fill(self, action):
        act_dict = self.actions[action][2]
        self._query.append([self.actions[action][0], act_dict['text'], act_dict['display_time'], act_dict['act_time']])

    def _query_clear(self, first_elem=None):
        if first_elem is None:
            self._query = []
        else:
            self._query = [first_elem]

    def _query_update(self):
        q = self._query.copy()
        for i in range(len(q) - 1, -1, -1):
            if q[i][3] == 0:
                self._do_now.append(q[i][0])
            if q[i][2] > 0:
                self._query[i][2] -= 1
                self._query[i][3] -= 1
            else:
                self._query.pop(i)

    def _do_actual(self):
        for elem in self._do_now:
            elem()
        self._do_now = []

    def _fake_act(self):
        pass

    def _default_1_act(self):
        self._query_clear([self._fake_act] + list(self.actions[1][2].values()))

    @staticmethod
    def _default_2_act():
        pass

    @staticmethod
    def _default_3_act():
        file_name = 'calc'
        os.system(file_name + '.exe')

    def _default_4_act(self):
        self._end_time = dt.datetime.now()

    def _default_5_act(self):
        self._resize_to -= 10
        self._resize_to = max(10, self._resize_to)

    def _default_6_act(self):
        self._resize_to += 10
        self._resize_to = min(300, self._resize_to)

    @staticmethod
    def _default_7_act():
        pass

    def _init_run(self):
        self._delete_imgs()
        self._end_time = None
        self._iteration = 0
        self._last_shown = 0
        self._query = []
        self._do_now = []
        self._started_at = dt.datetime.now()

    def _chain_res_builder(self, res):
        if self._res is None:
            self._res = res.clone().detach()
        else:
            self._res = torch.concat((self._res, res), dim=0)[-self._chain_delay:].clone().detach()

    def _res_calc(self):
        res = self._res.mean(dim=0)

        return res

    def _gesture_detect(self,
                        img: np.array,
                        transform: transforms.transforms.Compose,
                        model: torch.nn.Module):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img)
        res = F.softmax(model(img[None]), dim=1)
        self._chain_res_builder(res)
        res_final = self._res_calc()
        self.shown = res_final.argmax(dim=0).detach().numpy().tolist()
        self.shown_perc = res_final.max(dim=0).values.detach().numpy().tolist()

        return res_final, img

    def _gesture_pic_draw(self, frame, pos):
        x_offset, y_offset = pos
        small_img = cv2.resize(self._sign_pics[self.shown], (self._resize_to, self._resize_to))
        large_img = frame.copy()

        small_img[:, :, 0:3] = self._text_color

        rows, columns, chanels = small_img[:, :, :3].shape
        roi = large_img[y_offset:self._resize_to + y_offset,
                        x_offset:self._resize_to + x_offset]
        mask = 255 - small_img[:, :, 3]
        bg = cv2.bitwise_or(roi, roi, mask=mask)
        mask_inv = 255 - mask
        fg = cv2.bitwise_and(small_img[:, :, :3], small_img[:, :, :3], mask=mask_inv)
        final_roi = cv2.add(bg, fg)
        small_img = final_roi
        frame[y_offset:y_offset + small_img.shape[0],
              x_offset:x_offset + small_img.shape[1]] = small_img

        return frame

    def _put_text_shown(self, frame, pos):
        frame = frame.copy()
        if self._face_detected:
            text = f'{self._sign_dict[self.shown]} shown [{round(self.shown_perc * 100, 2)}%]'
        else:
            text = 'Face not found'
        cv2.putText(frame,
                    text,
                    pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA)
        cv2.putText(frame,
                    text,
                    pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    self._text_color,
                    1,
                    cv2.LINE_AA)

        return frame

    def _put_text_activity(self, frame, activity, pos):
        frame = frame.copy()
        cv2.putText(frame, activity, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, activity, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, self._text_color, 1, cv2.LINE_AA)

        return frame

    def _put_text_fps(self, frame, fps, pos):
        frame = frame.copy()
        cv2.putText(frame, f'FPS={round(fps, 1)}', pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'FPS={round(fps, 1)}', pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._text_color, 1, cv2.LINE_AA)

        return frame

    def _delete_imgs(self):
        [f.unlink() for f in self._RESULTS_PATH.glob('*.png')]

    def _face_detect(self, frame):
        face_detect = self._face_detect_model.detect(frame, landmarks=False)
        self._face_detected = False if face_detect[0] is None else True

    def _make_gif(self, duration):
        files = list(self._RESULTS_PATH.glob('*.png'))

        with imageio.get_writer(self._RESULTS_PATH / 'sign_vis.gif', mode='I',
                                duration=duration / len(files)) as writer:
            for file in sorted([f for f in files], key=lambda x: x.stat().st_ctime):
                image = imageio.imread(file)
                writer.append_data(image)
        self._delete_imgs()

    def run(self):
        self._init_run()

        cap = cv2.VideoCapture(0)
        try:
            while True:
                start = dt.datetime.now()
                _, frame = cap.read()
                frame = cv2.resize(frame, self._pic_size)
                if not self._iteration % self._face_detect_freq:
                    self._face_detect(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if self._face_detected:
                    res, img = self._gesture_detect(frame, self._transform, self._gesture_model)

                    if self.shown != 0:
                        if self.shown in self._cont_actions:
                            self._query_fill(self.shown)
                        elif self.shown in self._fixed_actions and self.shown != self._last_shown:
                            self._query_fill(self.shown)

                        frame = self._gesture_pic_draw(frame, (20, 70 + 20 * len(self._query)))

                    self._last_shown = self.shown

                self._query_update()

                for i, item in enumerate(self._query):
                    frame = self._put_text_activity(frame, item[1], pos=(20, 60 + 20 * i))

                frame = self._put_text_shown(frame, pos=(20, 30))

                self._do_actual()

                if self._end_time is not None and self._end_time < dt.datetime.now():
                    break

                clear_output(wait=True)

                try:
                    fps = 1 / (dt.datetime.now() - start).total_seconds()
                except ZeroDivisionError:
                    fps = 100500
                print(f'FPS = {fps}')
                frame = self._put_text_fps(frame, fps, pos=(10, 470))

                cv2.imshow('Normal Video', frame)

                if self._save_imgs:
                    cv2.imwrite(str(self._RESULTS_PATH / f'{self._iteration}.png'), frame)
                self._iteration += 1

        finally:
            duration = (dt.datetime.now() - self._started_at).total_seconds()
            cv2.destroyAllWindows()
            cap.release()
            if self._save_imgs:
                self._make_gif(duration)


new_o = GestureRecog(torch.load('models/sign_d_ep_15_acc_0.998', map_location=torch.device(device)),
                     pic_size=(640, 480),
                     save_imgs=False,
                     face_detect_freq=10,
                     chain_delay=10,
                     )

new_o.run()
