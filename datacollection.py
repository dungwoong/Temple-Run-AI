import matplotlib.pyplot as plt
import pyautogui as pg
import time
import keyboard


class CollectorError(Exception):
    pass


def find_screen():
    tlp = pg.locateOnScreen('res/topleft.jpg', confidence=0.9)
    if tlp is None:
        tlp = pg.locateOnScreen('res/topleft2.jpg', confidence=0.9)
    brp = pg.locateOnScreen('res/botright.jpg', confidence=0.9)
    if tlp is None or brp is None:
        raise CollectorError('Could not find phone screen.')
    left, top = (tlp.left, tlp.top)
    right, bot = (brp.left + brp.width, brp.top + brp.height)
    print(f"[FIND_SCREEN] {right - left}W x {bot - top}H")
    return {'l': left, 'r': right, 't': top, 'b': bot, 'h': (bot - top), 'w': (right - left)}


class CircularQueue:
    """
    A FIFO Queue with a set number of items.
    """

    def __init__(self, n):
        """Initializes a queue with length n"""
        self.n = n
        self.items = [None] * n
        self.size = 0
        self.start = 0

    def enqueue(self, item):
        if self.is_full():
            return
        self.items[self.get_end()] = item
        self.size += 1

    def dequeue(self):
        if self.is_empty():
            return
        tmp = self.items[self.start]
        self.items[self.start] = None
        self.start = (self.start + 1) % self.n
        self.size -= 1
        return tmp

    def is_full(self):
        # We could implement with self.start/end, but I think this is faster with more space used.
        return self.size == self.n

    def is_empty(self):
        return self.size == 0

    def get_end(self):
        return (self.start + self.size) % self.n


class KeyboardDetector:
    """
    collects real time data for now.
    """

    def get_state(self):
        # w = DOWN BY THE WAY
        state = ''
        if keyboard.is_pressed('up'):
            state += 'u'
        if keyboard.is_pressed('down'):
            state += 'w'
        if keyboard.is_pressed('left'):
            state += 'l'
        if keyboard.is_pressed('right'):
            state += 'r'
        if keyboard.is_pressed('a'):
            state += 'A'
        if keyboard.is_pressed('d'):
            state += 'D'
        return state


class DataCollector:
    def screenshot(self):
        """
        Collects a screenshot
        """
        raise NotImplementedError

    def get_game_state(self):
        """
        gets game state: which keys are pressed down
        """
        raise NotImplementedError

    def run(self, root='data'):
        """
        Performs a single step of the running process.
        This involves saving an image to a folder located at <root>
        """
        # look at current state of the keyboard
        # dequeue and save with label if queue is full.
        # collect()
        raise NotImplementedError


class RealTimeDataCollector(DataCollector):
    """
    Takes screenshots of the game in regular intervals.

    Regularly saves the screenshots, along with the accompanying label.
    """

    def __init__(self, last_id=0):
        # find the phone screen
        print('[DataCollector.INIT] waiting to find phone screen...')
        time.sleep(5)
        dims = find_screen()
        top = dims['t'] + dims['h'] // 4

        self.dims = (dims['l'], top, dims['r'], dims['b'])
        self.detector = KeyboardDetector()
        self.id = last_id

    def screenshot(self):
        """
        Takes a screenshot and crops it and returns it
        """
        img = pg.screenshot()
        img = img.crop(self.dims)
        img = img.resize((238, 310))
        return img

    def get_game_state(self):
        return self.detector.get_state()

    def run(self, root='data'):
        state = self.get_game_state()
        if len(state) == 0:
            state_str = 'na/'
        elif len(state) == 1:
            state_str = state + '/'
        else:
            state_str = 'mult/'
        im = self.screenshot()
        idx = '{:06d}'.format(self.id)
        fname = f'{root}/{state_str}{idx}.jpg'
        im.save(fname)
        self.id += 1

    def test_time(self, n=100, out_path='misc_img/datacollection_time.png'):
        times = []
        for i in range(n):
            start = time.time()
            img = self.screenshot()
            img.save('data/tmp.jpg')
            end = time.time()
            times.append(end - start)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.hist(times)
        ax1.set_xlabel('Time(s)')

        ax2.boxplot(times)
        ax2.set_xlabel('Time(s)')
        plt.suptitle(f'Time to screenshot and save model, n={n}, avg={sum(times) / len(times)}')
        plt.savefig(out_path)


if __name__ == '__main__':
    dc = RealTimeDataCollector()
    for i in range(2000):
        dc.run()
    print(dc.id)
