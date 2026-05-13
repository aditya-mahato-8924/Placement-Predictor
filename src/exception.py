from src.logger import logging
import sys

def error_message_details(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno

    error_message = f"""
    Error occurred in python script name [{file_name}]
    line number [{line_no}]
    error message [{str(error)}]
    """

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.error("Divide by zero error occurred.")
        raise CustomException(e, sys)