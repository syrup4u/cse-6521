from abc import abstractmethod, ABC, ABCMeta
from enum import Enum, EnumMeta

class AbstractStateMeta(ABCMeta, EnumMeta):
    pass

class AbstractState(ABC, Enum, metaclass=AbstractStateMeta):
    @property
    @abstractmethod
    def meaning(self) -> str:
        """Returns a human-readable description of the state's meaning."""
        pass

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
