import os
import sys
from loguru import logger


def setup_logger(log_dir="logs"):
    """
    篇日志文件
    :param log_dir: 日志目录
    """
    os.makedirs(log_dir,exist_ok=True)
    logger.remove()
    # 自定义终端处理器  {level:*<8}  <(左对齐) 8(加上level输出八个字符) *(除了level其它的用*占位)
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:*<8}</level> |"
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>-<level>{message}</level>",
        level="INFO"
    )
    # 添加文件处理器
    logger.add(
        os.path.join(log_dir,"doc_qa_{time:YYYY-MM-DD}.log"),
        rotation="00:00", # 每天轮换
        retention="30 days",  # 保留30天
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )
    return logger