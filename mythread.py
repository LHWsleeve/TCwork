from multiprocessing import Pool,Manager
import os, time, random

def long_time_task(name,n):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))
    d[name] = name
    return

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    manager=Manager()
    d = manager.dict()
    p = Pool(4)
    for i in range(5):
        a = p.apply_async(long_time_task, args=(i,d))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print(d)

    print('All subprocesses done.')