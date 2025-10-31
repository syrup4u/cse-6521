from abc import abstractmethod

class AbstractState:
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
