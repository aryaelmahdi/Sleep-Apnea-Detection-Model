from dataclasses import dataclass

@dataclass
class ModelResponse:
    code: int
    message: str
    result: str
    
    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "result": self.result
        }
    
@dataclass
class ExtensionNotAllowed:
    code: int = 400
    message: str = "Error! File not supported"

    def to_dict(self):
        return {
            "code":self.code,
            "message": self.message
        }