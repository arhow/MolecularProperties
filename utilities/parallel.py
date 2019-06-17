from multiprocessing import Pool, Value, Manager
from multiprocessing import Queue as PQueue
import multiprocessing
import numpy as np
import pandas as pd
import time
import sys


class Parallel(object):

    """
    generate samples from some data structure
    """

    def __init__(self, objective, objective_kwargs):
        self.objective = objective
        self.objective_kwargs = objective_kwargs
        return

    def run(self, iterable_obj, n_jobs=-1):

        try:
            iter(iterable_obj)
        except Exception as e:
            raise Exception(e.__str__())

        X = Manager().list()
        if n_jobs == -1:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        pool = Pool(n_jobs)

        params = []
        for item in iterable_obj:
            params.append((X, item))
        pool.starmap(self.worker, params)
        pool.close()
        pool.join()
        return list(X)


    def worker(self, X, item):
        try:
            d_ = self.objective(item, **self.objective_kwargs)
            assert type(d_)==dict, 'objective not return a dict object'
            X.append({**d_})
        except Exception as e:
            raise Exception()

"""

lst_ = [1,2,3,4]

def objective(item):
    return {'data':item + 3}

paralel = Parallel(objective)

df_ = pd.DataFrame(paralel.run(lst_))

"""