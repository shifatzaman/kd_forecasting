
from abc import ABC, abstractmethod

class TeacherModel(ABC):
    @abstractmethod
    def fit(self, series):
        pass

    @abstractmethod
    def predict(self, X):
        pass
