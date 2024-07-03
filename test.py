from collections.abc import Iterable
from typing import Any
import torch
import numpy as np


class Klass:
    def __init__(self) -> None:
        print('init')
        self.data='data'

    def __repr__(self) -> str:
        print('repr')
        return 'Klass: '+self.data
    
    def __eq__(self, value: object) -> bool:
        print('eq')
        if isinstance(value, Klass):
            return self.data==value.data
        return False
    
    def __hash__(self) -> int:
        print('hash')
        return hash(self.data)
    
    def __index__(self) -> int:
        print('index')
        return 0
    
    def __len__(self) -> int:
        print('len')

    def __iter__(self):
        print('iter')
        return iter([1,2,3])

    def __getitem__(self):
        print('getitem')

    def __setitem__(self):
        print('setitem')

    def __delitem__(self):
        print('delitem')

    def __contains__(self, value):
        print(f'contains {value}')

    def __next__(self):
        print('next')

    def __missing__(self,value):
        print(f'missing {value}')

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print(f'call {args} {kwds}')

    def __getattr__(self,key):
        print(f'getattr {key}')
        return f'getattr {key}'
    
    # def __getattribute__(self, name: str) -> Any:
    #     print(f'getattribute {name}')
    #     return f'getattribute {name}'
    
    def __setattr__(self, name: str, value: Any) -> None:
        print(f'setattr {name} {value}')

    def __delattr__(self, name: str) -> None:
        print(f'delattr {name}')

    def __dir__(self) -> Iterable[str]:
        print(f'dir')
    
klass=Klass()
for e in klass:
    print(e)
klass()
klass.attr=1
klass.data
print(dir(object))
print(dir(type))

# ws=torch.zeros((10,1), dtype=torch.float)
# pick=torch.tensor([[1,1],[1,1]], dtype=torch.int)
# print(ws[pick])
# ws[pick]+=torch.tensor([[2,2],[2,2]]).unsqueeze(-1)
# print(ws)
# ttt=torch.tensor([[1,1],[1,1]]).unsqueeze(-1)
# ttt+=torch.tensor([[2,3],[2,2]]).unsqueeze(-1)
# print(ttt)

ws=np.zeros((10,1), dtype=np.float32)
pick=np.array([[1,1],[1,1]], dtype=np.int32)
print(ws[pick])
ws[pick]+=np.array([[[4],[4]],[[3],[5]]])
print(ws)
t1=np.array([[0]])
pick=np.array([[0,0],[0,0]])
print(t1[pick])
t1[pick]+=np.array([[[1],[2]],[[3],[4]]])
print(t1)