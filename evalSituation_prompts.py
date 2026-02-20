import base64
import os
import re
import json
import time
from pathlib import Path
from typing import Tuple

import pyautogui
import pygetwindow as gw
from openai import OpenAI
from dotenv import load_dotenv


