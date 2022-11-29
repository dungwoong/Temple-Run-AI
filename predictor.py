import time
import torch
import pyautogui as pg
import numpy as np
import matplotlib.pyplot as plt

from datacollection import find_screen
from nets.shufflenet import ShuffleNetV2


class Predictor:
    def __init__(self, model, gpu=True):
        self.device = 'cuda' if torch.cuda.is_available() and gpu else 'cpu'
        print('[Predictor] Device =', self.device)
        self.model = model.to(self.device)
        print('[Predictor] Alert: setting model to eval()')
        self.model.eval()
        self.logsoftmax = torch.nn.LogSoftmax(dim=0).to(self.device)
        print('[Predictor] Waiting 2s to find phone screen: ')
        time.sleep(2)

        # get dims
        dims = find_screen()
        top = dims['t'] + dims['h'] // 4

        self.dims = (dims['l'], top, dims['r'], dims['b'])

        print('[Predictor] creating predict method for the model')
        self.create_predict()

    def predict(self, img):
        # assert img.shape[0] == 1, 'Must give singular observation.'
        with torch.no_grad():
            img = img.to(self.device)
            output = self.model(img)
            output = self.logsoftmax(output)
            output = torch.unsqueeze(output, dim=0)
            decision = torch.argmax(output)
            return decision.item()

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
        img = img.resize((238, 310))  # TODO hardcoded
        img = np.array(img)
        frame = torch.Tensor(img)
        # image is in BGR
        cropped = torch.transpose(frame, 2, 0)  # now the image is C x W x H.
        cropped = torch.unsqueeze(cropped, 0)
        return cropped

    def test_time(self, n=100, indiv_file='misc_img/predictor_indiv.png', total_file='misc_img/predictor_total.png'):
        """
        Testing method. Will make <n> predictions using the model, and record timing statistics.

        Saves statistics for total time, and individual screenshotting vs predicting times at respective file paths.
        """
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

            pred = self.predict(frame)
            print(frame.shape)
            print(pred)
            # print(time.time() - mid, 'to predict')
            end = time.time()
            pred_times.append(end - mid)
            total_times.append(end - start)
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


if __name__ == '__main__':
    m = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], num_classes=6)
    p = Predictor(m)
    p.test_time(100)
