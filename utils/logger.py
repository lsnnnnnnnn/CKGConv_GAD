import logging
import datetime
import os

def setup_logger(output_path: str, log_to_file=True, name="", add_date=True):
    if add_date:
        name_with_time = name + "_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    else:
        name_with_time = name

    os.makedirs(output_path, exist_ok=True)
    log_format = "%(asctime)s | %(name)s | %(levelname)s || %(message)s"

    handlers = [logging.StreamHandler()]
    if log_to_file:
        handlers.append(logging.FileHandler(f"{output_path}/{name_with_time}.log"))

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )

    logger = logging.getLogger(name)
    return logger
