from enum import Enum, EnumMeta


class EnumContainsMeta(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=EnumContainsMeta): ...
