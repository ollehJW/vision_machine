import json
import sys
from datetime import datetime
if (__package__=='') or (__package__ == None):
    from common_utils import CustomJSONEncoder
else:
    from .common_utils import CustomJSONEncoder

def get_log_idx(func):
    def wrapper(*args, **kwargs):
        wrapper.log_idx +=1
        return func(*args, **kwargs)
    wrapper.log_idx = -1
    return wrapper

class Logger(object):
    """Logging system of Sound Classifier module
    """
    def __init__(self, log_path) -> None:
        super().__init__()
        self._set_logger()
        self.log_path = log_path

    def warning(self, warning_message):
        self._basic_log("warning", warning_message)
    
    def error(self, error_message):
        self._basic_log("error", error_message)
        sys.exit()

    def info(self, info_message):
        self._basic_log("info", info_message)

    def result_info(self, update_part={}):
        self._log["result_info"].update(update_part)
        self._save()

    def _basic_log(self, log_level, message):
        message = message if isinstance(message, str) else str(message)
        message = self._get_nowtime() + str(message)
        self._log[log_level].append(message)
        self._save()

    def _set_logger(self):
        self._log = {"error":[],
                    "warning":[],
                    "info":[],
                    "result_info":{}
                    }

    def _save(self):
        with open(self.log_path, 'w') as f:
            json.dump(self._log, f, cls=CustomJSONEncoder)

    def _get_nowtime(self):
        nowtime = datetime.now()
        return nowtime.strftime("%Y-%m-%d-%H-%M-%S : ")

    def __repr__(self) -> str:
        return str(self._log)

    @property
    def log(self):
        return self.log

# log initialization in memory
_MODULE_LOG = []

@get_log_idx
def initiate_log(json_path):
    global _MODULE_LOG
    _MODULE_LOG.append(Logger(json_path))
    global _LOG_IDX
    _LOG_IDX = initiate_log.log_idx

def save(idx=None):
    idx = idx if idx else _LOG_IDX
    _MODULE_LOG[idx]._save()
def error(error_message,idx=None):
    idx = idx if idx is not None else _LOG_IDX
    _MODULE_LOG[idx].error(error_message)
def warn(warning_message,idx=None):
    idx = idx if idx is not None else _LOG_IDX
    _MODULE_LOG[idx].warning(warning_message)
def info(info_message,idx=None):
    idx = idx if idx is not None else _LOG_IDX
    _MODULE_LOG[idx].info(info_message)
def result_info(result_message, idx=None):
    idx = idx if idx is not None else _LOG_IDX
    _MODULE_LOG[idx].result_info(result_message)
def get_log(idx=None):
    idx = idx if idx is not None else _LOG_IDX
    return _MODULE_LOG[idx].log()

def flush():
    pass
    # global _LOG_IDX
    # _LOG_IDX -=1
    # del _MODULE_LOG[-1]

if __name__ =="__main__":
    from time import sleep
    def simple_log_test_function():
        for i in range(10):
            sleep(0.1)
            info(f"{i}")
        return i
    initiate_log("./test_log.log")
    simple_log_test_function()
    initiate_log("./test_log2.log")
    simple_log_test_function()
    initiate_log("./test_log3.log")
    simple_log_test_function()
    info('hey')
    warn("warning",idx=0)
    error("error", idx=1)
    warn('expected at log3')
    flush()
    warn('expected')