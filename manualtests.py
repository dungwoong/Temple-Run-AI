from datacollection import DataCollector, CollectorError


def test_no_phone():
    try:
        dc = DataCollector()
    except CollectorError:
        pass
