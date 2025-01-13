from abc import ABC, abstractmethod  # Abstract Base Class，用于定义抽象基类
from dataclasses import dataclass
from typing import Dict, Union, List


SLOT = Union[str, List[str], Dict[str, str]] # 类型别名(插槽)，可以是字符串、字符串列表、字符串字典之一
# 后续不同的template类定义中会传入不同的SLOT形式，以满足不同LLM的Prompt格式需求

# 每个Formatter都有一个slot属性，用于存储格式化字符串(模板)
# 每个Formatter的方法apply
@dataclass
class Formatter(ABC):
    slot: SLOT = ""

    @abstractmethod
    def apply(self, **kwargs) -> SLOT: ...


@dataclass
class EmptyFormatter(Formatter):
    def apply(self, **kwargs) -> SLOT:
        return self.slot
    #EmptyFormatter的apply方法直接返回slot，在Template中定义为一个列表(seperator)或字符串(system)，


@dataclass
class StringFormatter(Formatter):
    # apply方法接收一个字典kwargs，将其中的name(模板里定义的都是{{content}})替换为相应value
    def apply(self, **kwargs) -> SLOT:
        msg = ""
        for name, value in kwargs.items():
            if value is None:
                msg = self.slot.split(":")[0] + ":"
                return msg
            if not isinstance(value, str):
                raise RuntimeError("Expected a string, got {}".format(value))
            msg = self.slot.replace("{{" + name + "}}", value, 1)
        return msg
