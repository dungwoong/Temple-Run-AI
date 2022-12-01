import time

import keyboard
import torch
import pyautogui as pg
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from datacollection import find_screen
from nets.shufflenet import ShuffleNetV2


class Predictor:
    def __init__(self, model, pred_action=lambda x: x, gpu=True):
        self.pred_action = pred_action

        # set device
        self.device = 'cuda' if torch.cuda.is_available() and gpu else 'cpu'
        print('[Predictor] Device =', self.device)

        # configure model
        self.model = model.to(self.device)
        print('[Predictor] Alert: setting model to eval()')
        self.model.eval()

        # configure log / softmax layers
        self.logsoftmax = torch.nn.LogSoftmax(dim=1).to(self.device)
        self.softmax = torch.nn.Softmax(dim=1).to(self.device)
        print('[Predictor] Waiting 2s to find phone screen: ')
        time.sleep(2)

        # find screen
        dims = find_screen()

        self.dims = (dims['l'], dims['t'], dims['r'], dims['b'])

        print('[Predictor] creating predict method for the model')
        self.create_predict()

    def preprocess(self, img):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = preprocess(img)  # img.float() if img is tensor
        img = torch.transpose(img, 1, 2)  # C W H
        img = torch.unsqueeze(img, 0)
        return img

    def predict(self, img, probs=False):
        # assert img.shape[0] == 1, 'Must give singular observation.'
        img = self.preprocess(img)
        with torch.no_grad():
            img = img.to(self.device)
            output = self.model(img)
            # print(output.shape)
            # output = torch.unsqueeze(output, dim=0)
            if not probs:
                output = self.logsoftmax(output)
                decision = torch.argmax(output, dim=1)
                return decision.item()
            else:
                output = self.softmax(output)
                value, indices = torch.max(output, dim=1)
                # output = self.logsoftmax(output)
                return value.item(), indices.item()

    def create_predict(self):
        """
        Predict method for the model is created when the model is first asked to make a prediction.
        Thus, we will set this here.

        We may not need this if we are loading in a pretrained model, but currently I am testing with blank ones...?
        """
        im = self.take_screenshot()
        self.predict(im)

    def take_screenshot(self):
        img = pg.screenshot()
        img = img.crop(self.dims)
        img = img.resize((238, 412))  # TODO hardcoded
        # img = np.array(img)
        # frame = torch.Tensor(img)
        # # image is in BGR
        # cropped = torch.transpose(frame, 2, 0)  # now the image is C x W x H.
        # cropped = torch.unsqueeze(cropped, 0)
        return img

    def run(self):
        frame = self.take_screenshot()
        prob, pred = self.predict(frame, probs=True)
        return self.pred_action((prob, pred))

    def test_time(self, n=100, save_to_file=True,
                  indiv_file='misc_img/predictor_indiv.png', total_file='misc_img/predictor_total.png'):
        """
        Testing method. Will make <n> predictions using the model, and record timing statistics.

        Saves statistics for total time, and individual screenshotting vs predicting times at respective file paths.
        """
        label_dict = {0: 'neutral',
                      1: 'A key',
                      2: 'D key',
                      3: 'left',
                      4: 'right',
                      5: 'up',
                      6: 'down'}
        last_pred = None
        # take a screenshot
        screenshot_times = []
        pred_times = []
        total_times = []
        for i in range(n):
            # take screenshot
            start = time.time()
            frame = self.take_screenshot()
            mid = time.time()
            screenshot_times.append(mid - start)

            prob, pred = self.predict(frame, probs=True)
            # print(frame.shape)
            if pred != last_pred:  # and pred in [3, 4, 5, 6]:
                last_pred = pred
                print(label_dict[pred], '(', prob, '%)')
            # print(time.time() - mid, 'to predict')
            end = time.time()
            pred_times.append(end - mid)
            total_times.append(end - start)
        if save_to_file:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            ax1.hist(total_times)
            ax1.set_xlabel('Total Time(s)')
            plt.suptitle(f'Total Time to run, n={len(total_times)}, avg={sum(total_times) / len(total_times)}')
            ax2.boxplot(total_times)
            ax2.set_xlabel('Total Time(s)')
            plt.savefig(total_file)

            # plot prediction and screenshotting times.
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            ax1.boxplot(screenshot_times)
            ax1.set_title(f'Screenshot times, avg={sum(screenshot_times) / len(screenshot_times)}')
            ax1.set_xlabel('Time(s)')
            ax2.boxplot(pred_times)
            ax2.set_title(f'Pred times, avg={sum(pred_times) / len(pred_times)}')
            ax2.set_xlabel('Time(s)')
            plt.savefig(indiv_file)
        # run the model on the screenshot
        # model will predict log prob of each action
        # do a movement based on the results of the model.


class PerformGameAction:
    """
    Will perform game action, and print to screen what the action was.
    """

    def __init__(self, lean_cooldown=0.5):
        self.lean_delay = 0.5
        self.last_lean = -1
        self.last_pred = -1
        self.label_fun_dict = {0: self.neutral,
                               1: self.a_key,
                               2: self.d_key,
                               3: self.left,
                               4: self.right,
                               5: self.up,
                               6: self.down}
        self.label_dict = {0: 'neutral',
                           1: 'A key',
                           2: 'D key',
                           3: 'left',
                           4: 'right',
                           5: 'up',
                           6: 'down'}

    def up(self):
        pg.press('up')

    def down(self):
        pg.press('down')

    def right(self):
        pg.press('right')

    def left(self):
        pg.press('left')

    def a_key(self):
        t = time.time()
        if t - self.last_lean >= self.lean_delay:
            self.last_lean = t
            pg.keyDown('a')
        else:
            return False

    def d_key(self):
        t = time.time()
        if t - self.last_lean >= self.lean_delay:
            self.last_lean = t
            pg.keyDown('d')
        else:
            return False

    def neutral(self):
        self.last_lean = -1
        pg.keyUp('a')
        pg.keyUp('d')

    def __call__(self, out, log=True):
        prob, pred = out
        func = self.label_fun_dict[pred]
        r = func()
        if log and r is None:
            if self.last_pred != pred:
                self.last_pred = pred
                print(self.label_dict[pred], "(", round(prob*100, 2), "% )")


if __name__ == '__main__':
    # m = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], num_classes=7)  # 1.0
    m = ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], num_classes=7)  # 0.5
    net_path = 'nets/pretrained/03.pth'
    m.load_state_dict(torch.load('nets/pretrained/04.pth'))
    p = Predictor(m, pred_action=PerformGameAction())
    # p.test_time(2000, save_to_file=False)a
    print('running in 3')
    time.sleep(1)
    print('2')
    time.sleep(1)
    print('1')
    time.sleep(1)
    while True:
        if keyboard.is_pressed('alt') or keyboard.is_pressed('space'):
            print('ending...')
            break
        p.run()
