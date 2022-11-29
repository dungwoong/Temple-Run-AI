import matplotlib.pyplot as plt
import pyautogui as pg
import time
import keyboard
from dataset import rename_by_indices


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

    def clear(self):
        self.items = [None] * self.n
        self.size = 0
        self.start = 0


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


class TriggerDetector:
    def __init__(self, up, down, left, right, a_key, d_key, na):
        """
        insert callables for when any key is pressed
        """
        self.u, self.d, self.l, self.r, self.a_key, self.d_key, self.na = up, down, left, right, a_key, d_key, na
        self.up_pressed, self.down_pressed, self.left_pressed, self.right_pressed = False, False, False, False

    def get_game_state(self):
        args = None
        if keyboard.is_pressed('up') and not self.up_pressed:
            self.up_pressed = True
            args = self.up
        elif not keyboard.is_pressed('up'):
            self.up_pressed = False

        if keyboard.is_pressed('down') and not self.down_pressed:
            self.down_pressed = True
            args = self.down
        elif not keyboard.is_pressed('down'):
            self.down_pressed = False

        if keyboard.is_pressed('left') and not self.left_pressed:
            self.left_pressed = True
            args = self.left
        elif not keyboard.is_pressed('left'):
            self.left_pressed = False

        if keyboard.is_pressed('right') and not self.right_pressed:
            self.right_pressed = True
            args = self.right
        elif not keyboard.is_pressed('right'):
            self.right_pressed = False

        if keyboard.is_pressed('a'):
            args = self.a_key
        if keyboard.is_pressed('d'):
            args = self.d_key

        if args is None:
            args = self.na
        return args


class DataCollector:
    def __init__(self, last_id=0, detector=KeyboardDetector()):
        # find the phone screen
        print('[DataCollector.INIT] waiting to find phone screen...')
        time.sleep(5)
        dims = find_screen()

        self.dims = (dims['l'], dims['t'], dims['r'], dims['b'])
        self.detector = detector
        self.id = last_id

    def screenshot(self):
        """
        Collects a screenshot, crops and returns the image.
        """
        img = pg.screenshot()
        img = img.crop(self.dims)
        img = img.resize((238, 412))
        return img

    def get_game_state(self):
        """
        gets game state: which keys are pressed down
        """
        return self.detector.get_state()

    def run(self, root='data'):
        """
        Performs a single step of the running process.
        This involves saving an image to a folder located at <root>
        """
        # look at current state of the keyboard
        # dequeue and save with label if queue is full.
        # collect()
        raise NotImplementedError


class PassiveDataCollector(DataCollector):
    """
    Takes screenshots of the game in regular intervals.

    When no buttons are pressed, will take a screenshot after a certain amount of time.
    """

    def __init__(self, sleep_time=0.1, queue_size=8, last_id=0, detector=KeyboardDetector()):
        super(PassiveDataCollector, self).__init__(last_id, detector)
        self.q = CircularQueue(queue_size)  # take screenshot every 0.1, we save 1.0s of data
        self.sleep_time = sleep_time
        self.last_time = -1

    def run(self, root='data'):
        state = self.get_game_state()

        if len(state) == 0 and time.time() - self.last_time >= self.sleep_time * 10:
            # nothing is happening, start enqueuing
            im = self.screenshot()
            self.q.enqueue(im)
            self.last_time = time.time()

        elif len(state) != 0:
            print('clearing...')
            self.q.clear()
            time.sleep(3 * self.sleep_time)
            self.last_time = time.time()

        if self.q.is_full():
            img = self.q.dequeue()
            self.q.clear()
            # save image
            idx = '{:06d}'.format(self.id)
            fname = f'{root}/na/{idx}.jpg'
            print(fname)
            img.save(fname)
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


class TriggerDataCollector(DataCollector):
    """
    Collects one point of data when we press any key.
    """

    def __init__(self, max_count, last_id=0, na_mult=1):
        # detector = TriggerDetector('data/u', 'data/w', 'data/l', 'data/r', 'data/A', 'data/D', 'data/na')
        detector = KeyboardDetector()
        super(TriggerDataCollector, self).__init__(last_id, detector)
        self.counter = 0
        self.na_counter = 0
        self.na_max = na_mult
        self.max = max_count
        self.up_pressed, self.down_pressed, self.left_pressed, self.right_pressed = False, False, False, False

    def save(self, folder):
        idx = '{:06d}'.format(self.id)
        fname = f'{folder}/{idx}.jpg'
        self.img.save(fname)

    def _set_pressed(self, key):
        if key == 'u':
            self.up_pressed = True
        elif key == 'w':
            self.down_pressed = True
        elif key == 'l':
            self.left_pressed = True
        elif key == 'r':
            self.right_pressed = True

    def run(self, root='data'):
        state = self.get_game_state()
        im = self.screenshot()
        self.counter += 1
        if state == '':
            self.up_pressed, self.down_pressed, self.left_pressed, self.right_pressed = False, False, False, False
            self.counter = self.max
            return
        if state in ['A', 'D'] and self.counter < self.max:
            return
        elif (state == 'l' and self.left_pressed) or (state == 'r' and self.right_pressed) or \
                (state == 'u' and self.up_pressed) or (state == 'w' and self.down_pressed):
            # not initial press
            return

        # now we assume if we got a/d/na counter is >= max, lruw was not previously pressed
        if self.counter >= self.max:
            self.counter = 0
        if len(state) == 1:
            self._set_pressed(state)
            state_str = state + '/'
        else:
            state_str = 'mult/'
        idx = '{:06d}'.format(self.id)
        fname = f'{root}/{state_str}{idx}.jpg'
        print(state_str)
        im.save(fname)
        self.id += 1


if __name__ == '__main__':
    total = rename_by_indices()
    # dc = TriggerDataCollector(max_count=50, last_id=total, na_mult=4)
    dc = PassiveDataCollector(queue_size=1, last_id=total)
    while True:
        dc.run()
