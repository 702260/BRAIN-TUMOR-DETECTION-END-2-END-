import flask 
import pickle
from io import BytesIO
from torch import argmax, load
from torch import device as DEVICE
from torch.cuba import is_available
from torch.nn import sequential, Linear, SELU, Dropout, LogSigmoid
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.models import resnet50
import os
UPLOAD_FOLDER = os.path.join('static', 'photos')
app = flask.Flask(__name__, template_folder = 'templates')
