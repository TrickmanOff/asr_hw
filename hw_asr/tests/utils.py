import platform
import shutil
from contextlib import contextmanager
from time import sleep

from hw_asr.utils.parse_config import ConfigParser


@contextmanager
def clear_log_folder_after_use(config_parser: ConfigParser):
    # this context manager deletes the log folders whether the body was executed successfully or not
    try:
        yield config_parser
    finally:
        if platform.system() == "Windows":
            # Running unittest on windows results in a delete lock on the log directories just skip
            # this cleanup for windows and wait 1s to have a different experiment name.
            # (if you know how to fix it, you are welcome to make pull request)
            sleep(1)
        else:
            config_parser.run_storage.remove_run()
            shutil.rmtree(config_parser.log_dir)
