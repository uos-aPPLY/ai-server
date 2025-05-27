import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("fastapi.log"),
        logging.StreamHandler()  # 콘솔에도 출력
    ]
)
logger = logging.getLogger("app")