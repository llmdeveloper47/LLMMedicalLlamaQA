import logging

def logger_object():
    logger_name = 'GPTDatasetBuilding'
    logger = logging.getLogger(logger_name)
    
    # Check if logger is already initialized to avoid reconfiguration
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)  # Set the base logging level
        
        # File handler to log detailed information
        file_handler = logging.FileHandler('experiment.log', mode='a')  # Append mode
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        
        # Console handler for logging to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Only log INFO or higher to console
        console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger