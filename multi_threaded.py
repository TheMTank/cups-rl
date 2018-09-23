"""
Copied from
http://ai2thor.allenai.org/tutorials/examples
"""

import threading
import time
import ai2thor.controller

thread_count = 8

def run():
    env = ai2thor.controller.Controller()
    env.start()

    # 100 is an arbritary number
    for _ in range(100):
        t_start = time.time()
        env.reset('FloorPlan1')
        env.step({'action' : 'Initialize', 'gridSize' : 0.25})
        print('init time', time.time() - t_start)
        t_start_total = time.time()
        for _ in range(10):
            env.step({'action' : 'MoveAhead'})
            env.step({'action' : 'RotateRight'})
        total_time = time.time() - t_start_total
        print('total time', total_time, 20 / total_time, 'fps')

threads = [threading.Thread(target=run) for _ in range(thread_count)]
for t in threads:
    t.daemon = True
    t.start()
    time.sleep(1)

for t in threads:
    # calling join() in a loop/timeout to allow for Python 2.7
    # to be interrupted with SIGINT
    while t.isAlive():
        t.join(1)

print('done')
