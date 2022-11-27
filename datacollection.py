import pyautogui as pg
import time


class CollectorError(Exception):
    pass


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


class DataCollector:
    """
    Takes screenshots of the game in regular intervals.

    Regularly saves the screenshots, along with the accompanying label.
    """

    def __init__(self, n):
        # Initialize the buffer of screenshots
        self.buffer = CircularQueue(n)

        # find the phone screen
        print('[DataCollector.INIT] waiting to find phone screen...')
        time.sleep(2)
        self.dims = self._find_screen()

    def collect(self):
        """
        Collects a new piece of data, and attempts to enqueue it.
        """
        pass

    def run(self):
        # look at current state of the keyboard
        # dequeue and save with label if queue is full.
        # collect()
        pass

    @staticmethod
    def _find_screen():
        tlp = pg.locateOnScreen('res/topleft.jpg', confidence=0.9)
        if tlp is None:
            tlp = pg.locateOnScreen('res/topleft2.jpg', confidence=0.9)
        brp = pg.locateOnScreen('res/botright.jpg', confidence=0.9)
        if tlp is None or brp is None:
            raise CollectorError('Could not find phone screen.')
        left, top = (tlp.left, tlp.top)
        right, bot = (brp.left + brp.width, brp.top + brp.height)
        print(f"[FIND_SCREEN] {right - left}W x {bot - top}H")
        return {'l': left, 'r': right, 't': top, 'b': bot}


if __name__ == '__main__':
    dc = DataCollector(2)
