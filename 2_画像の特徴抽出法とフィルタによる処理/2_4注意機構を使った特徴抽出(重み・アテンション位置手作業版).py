###注意機構を使った特徴抽出
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt # 画像の表示に使用
# Googleドライブをマウント
from google.colab import drive
drive.mount('/content/drive')

###特徴空間への射影
# 画像の読み込み
img = Image.open('drive/MyDrive/python_image_recognition/data/cosmos.jpg')
plt.imshow(img)# 画像の表示

# NumPyを使うため画像をNumPy配列に変換
img = np.asarray(img, dtype='float32')

# 画像を特徴空間に射影
w = np.array([[ 0.0065, -0.0045, -0.0018,  0.0075,
                0.0095,  0.0075, -0.0026,  0.0022],
              [-0.0065,  0.0081,  0.0097, -0.0070,
               -0.0086, -0.0107,  0.0062, -0.0050],
              [ 0.0024, -0.0018,  0.0002,  0.0023,
                0.0017,  0.0021, -0.0017,  0.0016]])
#img(256, 256, 3)を(256,256, 8)に変換
#256行256列の要素が、RGBベクトルだったのを、
#256行256列の要素が、8次元の特徴ベクトルに変換
features = np.matmul(img, w)

###アテンションの計算

# アテンション計算用の特徴(クエリ)を画像から抽出
feature_white = features[50, 50]  #この位置(50,50)に白い花があると仮定
feature_pink = features[200, 200] #この位置にピンクの花があると仮定
#クエリは8個の要素を持つ1次元配列(8,1)


#アテンションの計算
#各画素位置における特徴ベクトルとクエリベクトルの内積を計算
#例えば、(34,23)の位置において、特徴ベクトルF(34,23)
#とクエリベクトルQ(anywhere)の内積を計算
#F(x,y) * Q in all x,y
atten_white = np.matmul(features, feature_white)
atten_pink = np.matmul(features, feature_pink)

# ソフトマックスの計算
atten_white = np.exp(atten_white) / np.sum(np.exp(atten_white))
atten_pink = np.exp(atten_pink) / np.sum(np.exp(atten_pink))

## 表示用に最大・最小値で正規化
#正規化しないと真っ黒か真っ白になる
atten_white = (atten_white - np.amin(atten_white)) / \
#\は行の継続を示す
    (np.amax(atten_white) - np.amin(atten_white))
atten_pink = (atten_pink - np.amin(atten_pink)) / \
    (np.amax(atten_pink) - np.amin(atten_pink))

# NumPy配列をPIL画像に変換
img_atten_white = Image.fromarray(
    (atten_white * 255).astype('uint8'))

img_atten_pink = Image.fromarray(
    (atten_pink * 255).astype('uint8'))

print('白のコスモスに対するアテンション')
display(img_atten_white)
print('ピンクのコスモスに対するアテンション')
display(img_atten_pink)