from abc import abstractmethod, ABC, ABCMeta
from enum import Enum, EnumMeta

class AbstractStateMeta(ABCMeta, EnumMeta):
    pass

class AbstractState(ABC, Enum, metaclass=AbstractStateMeta):
    @property
    @abstractmethod
    def is_initial(self) -> bool:
        """Indicates whether the state is an initial state."""
        pass

    @property
    @abstractmethod
    def is_final(self) -> bool:
        """Indicates whether the state is a final state."""
        pass

    @classmethod
    @abstractmethod
    def get_lost_state(cls) -> 'AbstractState':
        """Returns the state that represents a lost condition."""
        pass

    @classmethod
    @abstractmethod
    def get_initial_states(cls) -> list['AbstractState']:
        """Returns a list of all possible initial states."""
        pass

    @classmethod
    @abstractmethod
    def get_final_states(cls) -> list['AbstractState']:
        """Returns a list of all possible final states."""
        pass

class DummyState(AbstractState):
    """
    A dummy implementation of AbstractState for testing purposes.
    """

    Initial = 0
    Intermediate = 1
    Final = 2
    Lost = 3

    @property
    def is_initial(self) -> bool:
        return self is DummyState.Initial

    @property
    def is_final(self) -> bool:
        return self is DummyState.Final

    @classmethod
    def get_lost_state(cls) -> 'DummyState':
        return DummyState.Lost

    @classmethod
    def get_initial_states(cls) -> list['DummyState']:
        return [DummyState.Initial]
