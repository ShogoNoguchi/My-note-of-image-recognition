#第3.1節 学習と評価の基礎
#モジュールのインポート
import random                    # 乱数生成用のモジュール
from collections import deque      # 双方向キューのためのモジュール
from tqdm import tqdm            # プログレスバー表示用モジュール
import numpy as np                # NumPyライブラリ
from PIL import Image             # 画像処理ライブラリ
import matplotlib.pyplot as plt   # グラフ描画用モジュール
from sklearn.manifold import TSNE # t-SNEのためのモジュール

# PyTorch関係のモジュール
from torch.utils.data import Dataset, DataLoader      # データセットとデータローダーのためのモジュール
from torch.utils.data.sampler import SubsetRandomSampler  # サンプラーのためのモジュール
import torchvision                                   # CIFAR-10データセットのためのモジュール

#CIFAR-10データセットクラスの生成
#dataディレクトリにCIFAR-10データセットをダウンロード
# 訓練用データをダウンロード（True）するか、テスト用データをダウンロードするか（False）を指定
#CIFAR-10クラスのインスタンスを生成。これによってデータセットが使える。
dataset = torchvision.datasets.CIFAR10(
    root='data', train=True, download=True)


###各物体クラスの画像の表示
# 表示済みの画像のラベルを保存する変数
displayed_classes = set()
i = 0
# 全てのラベルの画像を1枚ずつ表示するまでループ
#このデータセットの場合はi < len(dataset) の条件は不要
while i < len(dataset) and \
      len(displayed_classes) < len(dataset.classes):
    # インデックスを使って1サンプルを取得
    img, label = dataset[i]
    if label not in displayed_classes:
        print(f'物体クラス: {dataset.classes[label]}')

        # 元画像が小さいので、リサイズして表示
        img = img.resize((256, 256))
        display(img)

        # 表示済みラベルの追加
        displayed_classes.add(label)

    i += 1

###t-SNE（次元削減アルゴリズム）によるデータ分布の可視化

# t-SNEのためにデータを整形
x = []
y = []
num_samples = 200
for i in range(num_samples):
    img, label = dataset[i]

    # 画像を平坦化 ([32, 32, 3] -> [3027]に変換)
    #これがflatten
    img_flatten = np.asarray(img).flatten()
    x.append(img_flatten)
    y.append(label)

#この状態ではxは3027要素のarrayが200個入ったリスト
#なので、これを(200,3072)のarrayに変換するのがnp.stack(x)の役割
#yは今リストなのでarrayに変える。
x = np.stack(x)   #x(200,3027) 200個の画像データ
y = np.array(y)   #y(200,) 200個のラベル
#こうすることで、それぞれが一つのarrayになり、t-SNEに入力できる形になる。


# t-SNEを適用
#sci-kit learnのTSNEクラスのインスタンスを生成. n_components=2で2次元に削減
#random_state=0で、毎回同じ結果が得られるようにする
t_sne = TSNE(n_components=2, random_state=0)
x_reduced = t_sne.fit_transform(x)

# 各ラベルの色とマーカーを設定。以下の関数はmatplotlibのデフォルトの色とマーカーを取得する関数
cmap = plt.get_cmap("tab10") #10色のカラーマップ「tab10」を取得
markers = ['4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D'] #マーカーの種類を指定

###データをプロット
plt.figure(figsize=(20, 15))#グラフのサイズを20x15の大きさに設定
#enumerate(dataset.classes)を使用して各クラス（cls）とそのインデックス（i）を取得
for i, cls in enumerate(dataset.classes):
    plt.scatter(x_reduced[y == i, 0], x_reduced[y == i, 1],
                c=[cmap(i / len(dataset.classes))],
                marker=markers[i], s=500, alpha=0.6, label=cls)
plt.axis('off')
plt.legend(bbox_to_anchor=(1, 1), fontsize=24, framealpha=0)
plt.show()

###データセットを分割
"""CIFAR10クラスでは、学習セットとテストセットは引数で分割可だが
検証セットは引数で分割不可なので、自分で分割する"""

#データセットを分割するための2つの排反なインデックス集合を生成する関数
'''
dataset    : 分割対象のデータセット
ratio      : 1つ目のセットに含めるデータ量の割合
random_seed: 分割結果を不変にするためのシード
'''
def generate_subset(dataset: Dataset, ratio: float,
                    random_seed: int=0):
    # サブセットの大きさを計算
    size = int(len(dataset) * ratio)
      #indicesは、データセット内の全インデックスを保持するリスト.
      #range(len(dataset))で、データセットの全インデックスを数字として生成し、それをリストとして保存"""
    indices = list(range(len(dataset)))

    # 二つのセットに分ける前にシャッフル
    random.seed(random_seed)
    random.shuffle(indices)

    # セット1とセット2のサンプルのインデックスに分割
    indices1, indices2 = indices[:size], indices[size:]

    return indices1, indices2

### 学習、テストセットの用意
train_dataset = torchvision.datasets.CIFAR10(
    root='data', train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(
    root='data', train=False, download=True)

# 学習セットのうち、検証セットに使う割合
val_ratio = 0.2

# Subsetの生成　#ここで生成されるのは、train_setとval_setのインデックスのリストであり、画像データそのものではない
val_set, train_set = generate_subset(train_dataset, val_ratio)

#画像データは、DataLoaderを使って取得する
#DataLoaderは下記のようにして生成する
"""train_loader = DataLoader(train_dataset, batch_size=64, sampler=SubsetRandomSampler(train_set))
   val_loader = DataLoader(train_dataset, batch_size=64, sampler=SubsetRandomSampler(val_set))
   test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)                               """


print(f'学習セットのサンプル数 : {len(train_set)}')
print(f'検証セットのサンプル数 : {len(val_set)}')
print(f'テストセットのサンプル数: {len(test_dataset)}')

###画像整形関数
'''
img         : 整形対象の画像
channel_mean: 各次元のデータセット全体の平均, [入力次元]➡flatten後（今なら3072）
channel_std : 各次元のデータセット全体の標準偏差, [入力次元]➡flatten後（今なら3072）
'''

#まずは各次元ごとの平均と標準偏差を計算する
'''
dataset: 平均と標準偏差を計算する対象のPyTorchのデータセット
'''
def get_dataset_statistics(dataset: Dataset):
    data = []
    for i in range(len(dataset)):
        # 3072次元のベクトルを取得(素手に画像がflattenされてること前提)
        img_flat = dataset[i][0]
        #通常dataset[i]は、(画像データ,ラベル)のタプルを返す。
        #dataset[i][0] はデータの部分（通常は画像）を取得するために使用.
        #dataset[i][1] とすれば、対応するラベルを取得。

        data.append(img_flat)
    # 第0軸を追加して第0軸でデータを連結
    data = np.stack(data)

    # データ全体の平均と標準偏差を計算
    #axis=0は列ごとに平均と標準偏差を計算することを意味する
    channel_mean = np.mean(data, axis=0)
    channel_std = np.std(data, axis=0)

    return channel_mean, channel_std





#画像をNumPy配列に変換し、正規化する関数
def transform(img: Image.Image, channel_mean: np.ndarray=None,
              channel_std: np.ndarray=None):
    # PILからNumPy配列に変換
    img = np.asarray(img, dtype='float32')

    # [32, 32, 3]の画像を3072次元のベクトルに平坦化
    x = img.flatten()

    # 各次元をデータセット全体の平均と標準偏差で正規化
    if channel_mean is not None and channel_std is not None:
        x = (x - channel_mean) / channel_std
#ここで、channel_meanとchannel_stdは,3072次元のarrayである。
#channel_mean is not None: この部分は、channel_mean に値が渡されているかを確認
    return x

#ラベルをOne-Hotベクトルに変換する関数
'''
label      : 物体クラスラベル
num_classes: データセットの物体クラス数
'''
def target_transform(label: int, num_classes: int=10):  #クラス数はデフォルトで10としておく
    # 数字 -> One-hotに変換
    y = np.identity(num_classes)[label]
#np.identity(num_classes)は、num_classes×num_classesの単位行列を生成する関数
#これによって、label番目の要素だけ1で他が0の、クラス数×クラス数の形のarrayが生成される

    return y

"""データ整形処理の確認"""
# 各次元のデータセット全体の平均と標準偏差を計算
#transformはさっき作った画像の正規化をする関数。
# しかし、まだ平均と標準偏差が計算されていないので、transform関数は平坦化のみ行う。
#これが、先ほど正規化処理の条件分岐でif not Noneとした訳。
dataset = torchvision.datasets.CIFAR10(
    root='data', train=True, download=True, transform=transform) #平坦化を含めたdatasetのインスタンス生成

#上記のインスタンス"datasetをget_dataset_statistics"関数に渡すことで、平均と標準偏差を計算
channel_mean, channel_std = get_dataset_statistics(dataset)

#正規化を含めた画像整形関数の用意
#ラムダ式を使って、引数xのみをtransform関数に渡す関数を生成
#ラムダ式によってchannel_mean, channel_stdは固定される。
img_transform = lambda x: transform(x, channel_mean, channel_std)

#整形関数を渡してデータセットクラスインスタンスを生成
dataset = torchvision.datasets.CIFAR10(
    root='data', train=True, download=True,
    transform=img_transform, target_transform=target_transform) 
#target_transformも、transformも引数は１つのみの状態。


"""img_transformをtransformパラメータに、target_transformをtarget_transformパラメータに渡して
    データセットクラスのインスタンスを生成しています。これにより、画像の読み込み時に自動的に正規化処理が適用されます。"""

#データのサンプルと表示
img, label = dataset[0]   #通常dataset[i]は、(画像データ,ラベル)のタプルを返す。
print(f'画像の形状: {img}')
print(f'ラベルの形状: {label}') 





########多クラスロジスティック回帰の実装


class MultiClassLogisticRegression:
    '''
    多クラスロジスティック回帰
    dim_input  : 入力次元
    num_classes: 分類対象の物体クラス数
    '''

    def __init__(self, dim_input: int, num_classes: int):
        # パラメータの初期化
        self.weight = np.random.normal(scale=0.01,
                                       size=(dim_input, num_classes))
        self.bias = np.zeros(num_classes)

    '''
    内部用ソフトマックス関数
    x: ロジット, [バッチサイズ, 物体クラス数]
    '''
    #各クラスの値を合計で割り、全てのクラスが足し合わせると1になるように調整します。
    def _softmax(self, x: np.ndarray):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    #axis = 1は、行ごとに計算することを意味する。keepdims=Trueは、次元数を保持することを意味する。
    '''
    物体クラスの確率を予測する関数
    x: 入力データ, [バッチサイズ, 入力次元]
    '''
    def predict(self, x: np.ndarray):
        # 入力に線形識別関数を適用
        y = np.matmul(x, self.weight) + self.bias 
        #x[バッチサイズ,入力次元]×weight[入力次元,物体クラス数] = y[バッチサイズ, 物体クラス数]
        #biasは元は[物体クラス数]だが、自動で[バッチサイズ, 物体クラス数]になる
        #➡各batchsizeごとに、同じbiasが加算される。
        y = self._softmax(y)

        return y

    '''
    パラメータを更新する関数
    x     : 入力データ,              [バッチサイズ, 入力次元]
    y     : One-hot表現されたラベル, [バッチサイズ, 物体クラス数]
    y_pred: 予測確率,                [バッチサイズ, 物体クラス数]
    lr    : 学習率
    '''
    def update_parameters(self, x: np.ndarray, y: np.ndarray,
                          y_pred: np.ndarray, lr: float=0.001):
        # 出力と正解の誤差を計算
        diffs = y_pred - y

        # 勾配を使ってパラメータを更新
        """各要素でWと次元を合わせてx(y_pred-y)を計算して、Wと要素ごとの引き算を行う"""
        self.weight -= lr * np.mean(x[:, :, np.newaxis] * diffs[:, np.newaxis,:], axis=0)
        #newaxisは、次元を追加する関数。diffs[:, np.newaxis,:]はdiffs[:,np.newaxis]と同じ意味。指定した次元に新たな次元を追加する。

        self.bias -= lr * np.mean(diffs, axis=0)
        #np.mean(..., axis=0) でバッチ方向に平均を取り、shapeを[入力次元, 物体クラス数]にする

    '''
    モデルを複製して返す関数
    '''
    def copy(self):
        model_copy = self.__class__(*self.weight.shape)
        model_copy.weight = self.weight.copy()
        model_copy.bias = self.bias.copy()

        return model_copy



"""モデル出力の確認"""

# モデルの生成、初期化
#MultiClassLogisticRegressionクラスのインスタンス(model)を生成
model = MultiClassLogisticRegression(32 * 32 * 3, 10)

# バッチサイズ1(1枚の画像データって意味)でランダムな入力を生成
x = np.random.normal(size=(1, 32 * 32 * 3))

# 予測
y = model.predict(x)

print(f'予測確率: {y[0]}')
#今、batchsize=1なので、y[0]にしか値が入っていない。
#y[batchsize,物体クラス数]の形になっているので、y[0]は、物体クラス数の長さのarrayになっている。



"""学習・評価の実装"""

class Config: #学習・評価におけるハイパーパラメータやオプションを管理するクラス
    '''
    ハイパーパラメータとオプションの設定
    '''
    def __init__(self):
        self.val_ratio = 0.2          # 検証に使う学習セット内のデータの割合
        self.num_epochs = 30          # 学習エポック数
        self.lrs = [1e-2, 1e-3, 1e-4] # 検証する学習率.具体的には、0.01、0.001、0.0001 という値です。
        self.moving_avg = 20          # 移動平均で計算する損失と正確度の値の数
        self.batch_size = 32          # バッチサイズ
        self.num_workers = 2          # データローダーに使うCPUプロセスの数

#学習と評価のための関数
def train_eval():
    config = Config()  #Configクラスのインスタンスを生成

    # 入力データ正規化のために学習セットのデータを使って
    # 各次元の平均と標準偏差を計算
    #ここのtransformは、画像の正規化を行う関数で、上の方で自分で定義したもの。このようにtransform引数に関数を渡すことで、画像の前処理をしてくれる。
    dataset = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True, transform=transform)
    #root='data' は、データセットが保存されるディレクトリを指定する引数

    #get_dataset_statisticsも、上の方で自分で定義した関数
    channel_mean, channel_std = get_dataset_statistics(dataset)

    # 正規化を含めた画像整形関数の用意
    #ラムダ式を使って、引数xのみをtransform関数に渡す関数を生成(channel_mean, channel_stdを固定)
    img_transform = lambda x: transform(x, channel_mean, channel_std)


    # 学習、評価セットの用意
    train_dataset = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True,
        transform=img_transform, target_transform=target_transform)
    #img_transformはラムダ式のtransform関数. target_transformも先ほど自分で定義した関数
    test_dataset = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True,
        transform=img_transform, target_transform=target_transform)

    # 学習・検証セットへ分割するためのインデックス集合の生成
    #generate_subsetも自分で定義したデータセットを分割するための2つの排反なインデックス集合を生成する関数
    val_set, train_set = generate_subset(
        train_dataset, config.val_ratio)

    print(f'学習セットのサンプル数　: {len(train_set)}')
    print(f'検証セットのサンプル数　: {len(val_set)}')
    print(f'テストセットのサンプル数: {len(test_dataset)}')



       #"""  ↑↑ここまでで、インデックスの分割まではできた。次は、DataLoaderを使って画像データを取得する。"""  


    # インデックス集合から無作為にインデックスをサンプルするサンプラー
    #SubsetRandomSamplerクラスはコンストラクタに渡されたインデックス集合から無作為にインデックスをサンプルするサンプラー
    train_sampler = SubsetRandomSampler(train_set)
    """torch.utils.data.sampler モジュールの SubsetRandomSampler クラスを使っています。
       SubsetRandomSampler: 指定されたインデックス集合（train_set）からランダムにサンプリングを行うサンプラーです。
       指定されたインデックスのみに基づいてデータを取得し、各エポックごとにランダムな順序でデータを返します。"""
    
    # DataLoaderを生成
    """torch.utils.data モジュールの DataLoader クラスを使っています。
    DataLoader クラスは、データをバッチ単位で取り出し、モデルに供給するために使用します。"""
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, sampler=train_sampler)
    """num_workers=config.num_workers: データをロードする際の並列プロセス数を指定します。
       num_workers の値が大きいほどデータロードが並列化され、高速化されます。ただし、CPUのコア数やメモリ容量に依存します。
       sampler=train_sampler: データのサンプリング方法を指定します。ここでは前の行で定義した train_sampler を使用しており、
       インデックスリスト train_set に基づいてランダムにデータをサンプリングするよう設定されています。"""
    val_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, sampler=val_set)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers)

    # 検証セットの結果による最良モデルの保存用変数
    val_loss_best = float('inf') #検証セットの損失が最小になるモデルを保存する変数
    model_best = None #目的関数が最小になるモデルを保存する変数 
    for lr in config.lrs:  #学習率を変えて学習を行う
        print(f'学習率: {lr}')

        # 多クラスロジスティック回帰モデルの生成
        model = MultiClassLogisticRegression(
            32 * 32 * 3, len(train_dataset.classes))
        
        #第2引数: len(train_dataset.classes):この引数は、分類対象の物体クラスの数を指定しています。

'''
data_loader: 評価に使うデータを読み込むデータローダ
model      : 評価対象のモデル
'''
def evaluate(data_loader: DataLoader,
             model: MultiClassLogisticRegression):
    losses = []
    preds = []
    for x, y in data_loader:
        x = x.numpy()
        y = y.numpy()

        y_pred = model.predict(x)

        losses.append(np.sum(-y * np.log(y_pred), axis=1))

        preds.append(np.argmax(y_pred, axis=1) == \
                     np.argmax(y, axis=1))

    loss = np.mean(np.concatenate(losses))
    accuracy = np.mean(np.concatenate(preds))

    return loss, accuracy


"""ーーーーーー学習の開始ーーーーーー"""

for epoch in range(config.num_epochs):
            with tqdm(train_loader) as pbar:
                pbar.set_description(f'[エポック {epoch + 1}]')

                """tqdm は進捗バーを表示するためのライブラリです。
                   train_loader は学習データをバッチごとに取得するための PyTorch の DataLoader です。
                   pbar として進捗バーを使って学習の進行状況を表示します。"""
                """進捗バーの表示に現在のエポック番号を設定します。エポックは1から始まるため、 epoch(初期0) + 1 となっています。"""

                # 移動平均計算用
                losses = deque()
                accs = deque()
                for x, y in pbar:
                    # サンプルしたデータはPyTorchのTensorに
                    # 変換されているのためNumPyデータに戻す
                    x = x.numpy()
                    y = y.numpy()

                    y_pred = model.predict(x)

                    # 学習データに対する目的関数と正確度を計算

                    #クロスエントロピー損失(Loss)の計算
                    loss = np.mean(
                        #axis=1 で、クラス次元に沿って合計を取ることで、各サンプルに対して1つの損失値が得られます。
                        np.sum(-y * np.log(y_pred), axis=1))
                    

                    #argmaxのインデックスは数字表現のクラスラベル

                    """"np.argmax(y_pred, axis=1) は、y_pred 配列の各行で最大値を持つインデックスを返します。
                        ここでの axis=1 は、行ごとに処理を行うことを意味します。
                        つまり、y_pred の各サンプルにおいて、最も確率が高いクラスのインデックスを取得します。
                        このインデックスがモデルの予測したクラスを示します。"""
                    
                    """np.argmax(y, axis=1) は、One-Hot表現の配列に対して各行で最大値（1）を持つインデックスを返します。
                       このインデックスは実際のクラスラベルを示します。"""
                    
                    """np.argmax(y_pred, axis=1) == np.argmax(y, axis=1) では、モデルが予測したクラスと実際のクラスを比較します。
                       各サンプルについて、予測したクラスが正しい場合は True、間違っている場合は False を返します。
                       この比較結果はブール値の配列になり、例えば [True, False, True, ...] というようになります。

                        np.mean(...):
                       ブール値の配列を np.mean() に渡すと、True は 1、False は 0 として計算されます。
                       つまり、この配列における True の割合、すなわち正しく分類されたサンプルの割合を計算します。
                       これが 正確度（Accuracy） です。正確度は、バッチ全体に対するモデルの予測の正確さを表し、0から1の値を取ります。"""
                    
                    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))

                    # 移動平均を計算して表示
                    losses.append(loss)
                    accs.append(accuracy)
                    if len(losses) > config.moving_avg:
                        losses.popleft()
                        accs.popleft()
                    pbar.set_postfix({'loss': np.mean(losses),
                                      'accuracy': np.mean(accs)})

                    # パラメータを更新
                    model.update_parameters(x, y, y_pred, lr=lr)


            # 検証セットを使って精度評価
            val_loss, val_accuracy = evaluate(val_loader, model)
            print(f'検証: loss = {val_loss:.3f}, '
                  f'accuracy = {val_accuracy:.3f}')

            # より良い検証結果が得られた場合、モデルを記録
            if val_loss < val_loss_best:
                val_loss_best = val_loss
                model_best = model.copy()

    # テスト
    test_loss, test_accuracy = evaluate(test_loader, model_best)
print(f'テスト: loss = {test_loss:.3f}, '
      f'accuracy = {test_accuracy:.3f}')


#学習の実行
train_eval()



