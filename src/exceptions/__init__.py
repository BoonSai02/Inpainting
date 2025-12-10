class CustomException(Exception):
    def __init__(self, message: str, sys_error: str = None):
        super().__init__(message)
        self.message = message
        self.sys_error = sys_error
        
    def __str__(self):
        if self.sys_error:
            return f"{self.message} (System Error: {self.sys_error})"
        return self.message
