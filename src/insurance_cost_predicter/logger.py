import os
import time
import logging

os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts/models", exist_ok=True)
os.makedirs("artifacts/visuals", exist_ok=True)

logging.basicConfig(
    filename="logs/loan_model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.Formatter.converter = time.localtime
logger = logging.getLogger("InsuranceCostPipeline")  