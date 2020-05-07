from time import clock, time, perf_counter
from functools import reduce


def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
           reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                  [(t * 1000,), 1000, 60, 60])


def now():
    return secondsToStr(perf_counter())


class Timing:
    line = "=" * 40

    def __init__(self, name):
        self.name = name
        self.log("Starting Test: "+name)
        self.start = perf_counter()

    def log(self, s, elapsed=None):
        print(self.line)
        print(secondsToStr(perf_counter()), '-', s)
        if elapsed:
            print("Elapsed time:", elapsed)
        print(self.line)
        print

    def end_log(self):
        end = perf_counter()
        elapsed = end - self.start
        self.log("End Program", secondsToStr(elapsed))

#timer = Timing()
#timer.start = clock()
#atexit.register(timer.end_log)
#timer.log("Start Program")
