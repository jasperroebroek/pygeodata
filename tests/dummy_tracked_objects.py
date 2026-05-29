from pygeodata.tracked_object import TrackedObject


class TrackedBase(TrackedObject):
    pass


class TrackedChild(TrackedBase):
    pass


class Foo(TrackedObject):
    pass


class Bar(TrackedObject):
    pass


class SimpleTrackedObject(TrackedObject):
    pass


class C(TrackedObject):
    def foo(self) -> None:
        D()


class D(TrackedObject):
    def bar(self) -> None:
        C()


class DuplicateTracked(TrackedObject):
    pass
