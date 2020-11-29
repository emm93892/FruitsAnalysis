import pathlib

import numpy as np
from keras.models import load_model
from PIL import Image
import io
from pathlib import Path

imsize = (32, 32)

def load_image(image):
    inst = io.BytesIO(Path.read_bytes(image))
    img = Image.open(inst)
    img = img.convert('RGB')
    # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
    img = img.resize(imsize)
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img / 255.0

    model = load_model("./cnn.h5")
    prd = model.predict(np.array([img]))
    #print(prd) # 精度の表示
    prelabel = np.argmax(prd, axis=1)

    return prelabel
