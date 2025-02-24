from transformers import logging

# Set a global verbosity level (INFO, DEBUG, etc.)
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# Replace print statements with logger calls
logger.info("Starting training ...") 