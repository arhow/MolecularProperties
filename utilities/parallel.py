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

        self.X = Manager().list()
        if n_jobs == -1:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        pool = Pool(n_jobs)

        params = []
        for item in iterable_obj:
            params.append((item,))
        pool.starmap(self.worker, params)
        pool.close()
        pool.join()
        return list(self.X)


    def worker(self, item):
        try:
            d_ = self.objective(item, **self.objective_kwargs)
            
            if type(d_) == dict:
                self.X.append(d_)
            elif type(d_) == list:
                for d_i_ in d_:
                    self.X.append(d_i_)
            else:
                'objective not return a dict object or a dict list'
        except Exception as e:
            raise Exception()

"""
a = [1,2,3,4,5,6,7,8,9,10]

df_info = pd.DataFrame({'a':[1,2,3], 'b':[5,6,7]})

def func_b_sub(a, b, c, df):
    return a + b + c + df.shape[0]*10

def func_b(item,  cc, **kwargs):
    id_ = item[0]
    x_ = item[1]
    return {'b':x_ + 10 + func_b_sub(**kwargs), 'idx':id_}

r = []
for item in enumerate(a):
    b = func_b(item, 10, a=100,b=100,c=100, df=df_info)
    r.append(b)
print(r)

r = Parallel(func_b,{'cc':10,'a':100,'b':100,'c':100, 'df':df_info}).run(enumerate(a))
print(r)
"""