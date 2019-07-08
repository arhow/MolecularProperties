import os
import sys
from utilities.process.pnode import PNode


class PQueue(object):

    def __init__(self, df_train, df_test, param, trial):
        self.root = PNode()
        self.last_pnode_point = self.root
        self.df_train = df_train
        self.df_test = df_test
        self.param = param.copy()
        self.trial = trial
        return

    def insert_node(self, pnode):
        self.last_pnode_point.next = pnode
        self.last_pnode_point = pnode

    def run(self):
        self._run_pnode(self.root)
        return

    def _run_pnode(self, pnode, **kwargs):
        next_node_kwargs = pnode.run(self.df_train, self.df_test, self.param, self.trial, **kwargs)
        if type(pnode.next)==type(None):
            return
        else:
            return self._run_pnode(pnode.next, **next_node_kwargs)





