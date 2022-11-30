import pytest
from datacollection import CircularQueue


def test_cq_empty():
    q = CircularQueue(5)
    assert q.dequeue() is None
    assert q.is_empty()


def test_cq_basic():
    q = CircularQueue(5)
    assert q.size == 0
    q.enqueue(1)
    assert q.size == 1
    q.enqueue(2)
    assert q.dequeue() == 1
    assert q.dequeue() == 2
    assert q.is_empty()


def test_full_q():
    q = CircularQueue(4)
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    q.enqueue(4)
    q.enqueue(5) # will not enqueue
    assert q.dequeue() == 1
    assert q.dequeue() == 2
    assert q.dequeue() == 3
    q.enqueue(6)
    assert q.dequeue() == 4
    assert q.dequeue() == 6
    assert q.dequeue() is None


if __name__ == '__main__':
    pytest.main(['tests.py'])
