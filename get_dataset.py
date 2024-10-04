import os
from roboflow import Roboflow


ROBOFLOW_API_KEY = input('Enter ROBOFLOW_API_KEY secret value: ')
# api key = uqTXlACGPeIn0KqpFHb7

os.makedirs(os.path.expanduser('~/datasets'), exist_ok=True)
os.chdir(os.path.expanduser('~/datasets'))

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("visually-impaired-obstacle-detection-uxdze").project("obstacle-detection-yeuzf")
version = project.version(11)
dataset = version.download("coco")

print(dataset.location)