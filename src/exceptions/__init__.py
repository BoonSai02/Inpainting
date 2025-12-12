import sys

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, message: str, sys_error: str = None, error_detail: sys = sys):
        super().__init__(message)
        self.message = message
        self.sys_error = sys_error
        
        # If there's an active exception, capture details
        if sys.exc_info()[2] is not None:
             self.detailed_message = error_message_detail(message if sys_error is None else sys_error, error_detail=error_detail)
        else:
             self.detailed_message = message

    def __str__(self):
        base_msg = self.detailed_message
        if self.sys_error and "System Error" not in base_msg:
             return f"{base_msg} (System Error: {self.sys_error})"
        return base_msg
