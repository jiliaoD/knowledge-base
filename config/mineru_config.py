from dataclasses import dataclass
import os
from dotenv import load_dotenv
load_dotenv(override=True)


@dataclass()
class MinerUConfig:
    base_url:str
    api_token:str
mineru_config=  MinerUConfig(
    base_url=os.getenv("MINERU_BASE_URL",""),
    api_token=os.getenv("MINERU_API_TOKEN","")
)
