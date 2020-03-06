from typing import TypeVar, Generic, Type, Callable, overload, Any, List

TParam = TypeVar("TParam")
TParam1 = TypeVar("TParam1")
TParam2 = TypeVar("TParam2")
TParam3 = TypeVar("TParam3")
TParam4 = TypeVar("TParam4")
TParam5 = TypeVar("TParam5")
TParam6 = TypeVar("TParam6")


class SignalConnection:
    _parent_signal: 'SignalBase'
    _cb: Callable[..., Any]
    enabled: bool

    def __init__(self, cb: Callable[..., Any], parent_signal: 'SignalBase'):
        self._parent_signal = parent_signal
        self._cb = cb
        self.enabled = True

    def disconnect(self):
        self._parent_signal.disconnect(self)

    def emit(self, *args, **kwargs):
        if self.enabled:
            self._cb(*args, **kwargs)


class SignalBase:
    _connections: List[SignalConnection]

    def __init__(self):
        self._connections = []

    def emit(self, *args, **kwargs):
        for cb in self._connections:
            cb.emit(*args, **kwargs)

    def connect(self, cb: Callable) -> SignalConnection:
        connection = SignalConnection(cb, self)
        self._connections.append(connection)
        return connection

    def disconnect(self, connection: SignalConnection):
        self._connections.remove(connection)


class Signal0(SignalBase):

    def emit(self):
        super().emit()

    def connect(self, cb: Callable[[], None]) -> SignalConnection:
        return super().connect(cb)


class Signal1(SignalBase, Generic[TParam1]):
    def __init__(self, p1: Type[TParam1]):
        super().__init__()

    def emit(self, p1: TParam1):
        super().emit(p1)

    def connect(self, cb: Callable[[TParam1], None]) -> SignalConnection:
        return super().connect(cb)


class Signal2(SignalBase, Generic[TParam1, TParam2]):
    def __init__(self, p1: Type[TParam1], p2: Type[TParam2]):
        super().__init__()

    def emit(self, p1: TParam1, p2: TParam2):
        super().emit(p1, p2)

    def connect(self, cb: Callable[[TParam1, TParam2], None]) -> SignalConnection:
        return super().connect(cb)


class Signal3(SignalBase, Generic[TParam1, TParam2, TParam3]):
    def __init__(self, p1: Type[TParam1], p2: Type[TParam2], p3: Type[TParam3]):
        super().__init__()

    def emit(self, p1: TParam1, p2: TParam2, p3: TParam3):
        super().emit(p1, p2, p3)

    def connect(self, cb: Callable[[TParam1, TParam2, TParam3], None]) -> SignalConnection:
        return super().connect(cb)


class Signal4(SignalBase, Generic[TParam1, TParam2, TParam3, TParam4]):
    def __init__(self, p1: Type[TParam1], p2: Type[TParam2], p3: Type[TParam3], p4: Type[TParam4]):
        super().__init__()

    def emit(self, p1: TParam1, p2: TParam2, p3: TParam3, p4: TParam4):
        super().emit(p1, p2, p3, p4)

    def connect(self, cb: Callable[[TParam1, TParam2, TParam3, TParam4], None]) -> SignalConnection:
        return super().connect(cb)


class Signal5(SignalBase, Generic[TParam1, TParam2, TParam3, TParam4, TParam5]):
    def __init__(self, p1: Type[TParam1], p2: Type[TParam2], p3: Type[TParam3], p4: Type[TParam4], p5: Type[TParam5]):
        super().__init__()

    def emit(self, p1: TParam1, p2: TParam2, p3: TParam3, p4: TParam4, p5: TParam5):
        super().emit(p1, p2, p3, p4, p5)

    def connect(self, cb: Callable[[TParam1, TParam2, TParam3, TParam4, TParam5], None]) -> SignalConnection:
        return super().connect(cb)


class Signal6(SignalBase, Generic[TParam1, TParam2, TParam3, TParam4, TParam5, TParam6]):
    def __init__(self, p1: Type[TParam1], p2: Type[TParam2], p3: Type[TParam3], p4: Type[TParam4], p5: Type[TParam5],
                 p6: Type[TParam6]):
        super().__init__()

    def emit(self, p1: TParam1, p2: TParam2, p3: TParam3, p4: TParam4, p5: TParam5, p6: TParam6):
        super().emit(p1, p2, p3, p4, p5, p6)

    def connect(self, cb: Callable[[TParam1, TParam2, TParam3, TParam4, TParam5, TParam6], None]) -> SignalConnection:
        return super().connect(cb)


# class Signals(Generic[TParam1, TParam2, TParam3]):
@overload
def signal() -> Signal0:
    pass


@overload
def signal(p1: Type[TParam1]) -> Signal1[TParam1]:
    pass


@overload
def signal(p1: Type[TParam1], p2: Type[TParam2]) -> Signal2[TParam1, TParam2]:
    pass


@overload
def signal(p1: Type[TParam1], p2: Type[TParam2], p3: Type[TParam3]) -> Signal3[TParam1, TParam2, TParam3]:
    pass


@overload
def signal(p1: Type[TParam1], p2: Type[TParam2], p3: Type[TParam3], p4: Type[TParam4]) -> Signal4[
    TParam1, TParam2, TParam3, TParam4]:
    pass


@overload
def signal(p1: Type[TParam1], p2: Type[TParam2], p3: Type[TParam3], p4: Type[TParam4], p5: Type[TParam5]) -> Signal5[
    TParam1, TParam2, TParam3, TParam4, TParam5]:
    pass


@overload
def signal(p1: Type[TParam1], p2: Type[TParam2], p3: Type[TParam3], p4: Type[TParam4], p5: Type[TParam5],
           p6: Type[TParam6]) -> Signal6[TParam1, TParam2, TParam3, TParam4, TParam5, TParam6]:
    pass


def signal(*args):
    count = len(args)
    t = [Signal0, Signal1, Signal2, Signal3, Signal4, Signal5, Signal6]
    if count >= len(t):
        raise ValueError("Signals up to 6 parameters are supported. This can be extended by "
                         "implementing Signal7, ... classes")
    return t[count](*args)
