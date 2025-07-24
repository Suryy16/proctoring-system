import os

# Environment configuration
IS_DOCKER = os.getenv('IS_DOCKER', 'false').lower() == 'true'
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
WEBSOCKET_HOST = os.getenv('WEBSOCKET_HOST', 'localhost')
API_HOST = os.getenv('API_HOST', 'localhost')
ROOT_DATABASE_DIR = os.getenv('ROOT_DATABASE_DIR', 'database')
API_BASE_URL = f"http://{API_HOST}:5000"