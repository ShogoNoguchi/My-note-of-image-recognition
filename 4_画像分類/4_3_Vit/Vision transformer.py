"""モジュールのインポートとGoogleドライブのマウント"""
from collections import deque
import copy
from tqdm import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as T

# Googleドライブをマウント
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('drive/MyDrive/python_image_recognition/4_classification/4_3_transformer')

import util
import eval





#まずはVisionTransformerのための各モジュールを実装していきます。構造を理解するためには、
# まずは一番下の方のVisionTransformerのクラスから参照しましょう。




"""マルチヘッドアテンションを使った自己アテンションの実装"""

#PyTorch でモデルや層（レイヤー）を定義する際には、通常この nn.Module を継承して新しいクラスを作ります。
# 例えば、ニューラルネットワーク全体やその一部（畳み込み層、全結合層、アテンション層など）を定義するのに使われます。
class SelfAttention(nn.Module):
    '''
    自己アテンション
    dim_hidden: 入力特徴量の次元
    num_heads : マルチヘッドアテンションのヘッド数
    qkv_bias  : クエリなどを生成する全結合層のバイアスの有無
    入力:x: [バッチサイズ, 特徴量数（num_patches + 1）, 特徴量次元(num_hidden)]
    '''
    def __init__(self, dim_hidden: int, num_heads: int,
                 qkv_bias: bool=False):
        super().__init__()
#super().__init__()により、親クラスのコンストラクタを呼び出し、親クラスの機能を継承します。


        # 特徴量を各ヘッドのために分割するので、
        # 特徴量次元をヘッド数で割り切れるか検証
        assert dim_hidden % num_heads == 0

        self.num_heads = num_heads  #これにより、クラス内の他のメソッドでこの変数を参照できるようになります。

        # ヘッド毎の特徴量次元
        dim_head = dim_hidden // num_heads

        # ソフトマックスのスケール値
        self.scale = dim_head ** -0.5

        # ヘッド毎にクエリ、キーおよびバリューを生成するための全結合層
        """
        クエリ、キー、バリューは同じ入力から得られるものであり、
        計算効率を上げるためにこれらを一回の変換で同時に求めることが一般的です。
        具体的には、入力ベクトルに対して一回の全結合層（線形変換）を適用し、出力を dim_hidden * 3 の次元にします。
        これにより、出力は一つの大きなベクトルとして扱われ、そのベクトルを三等分してクエリ、キー、バリューを取り出す
        ことが可能になります。
         """

        self.proj_in = nn.Linear(
            dim_hidden, dim_hidden * 3, bias=qkv_bias)

        # 各ヘッドから得られた特徴量を一つにまとめる全結合層
        self.proj_out = nn.Linear(dim_hidden, dim_hidden)

    '''
    順伝播関数
    x: [バッチサイズ, 特徴量数（num_patches + 1）, 特徴量次元(num_hidden)]
    '''
    def forward(self, x: torch.Tensor):        
        bs, ns = x.shape[:2]
        #x.shape[:2]というスライシングは、x.shapeの最初の2つの要素（次元）を取得するための書き方です。
        #バッチサイズ (bs) と特徴量数 (ns) を取り出しています。


        qkv = self.proj_in(x)
       #入力テンソル x に対して全結合層 proj_in を適用し、クエリ、キー、バリューを一度に生成します。
       # この結果、qkv の次元は [バッチサイズ, 特徴量数, dim_hidden * 3] となります。





        # view関数により
        # [バッチサイズ, 特徴量数, QKV, ヘッド数, ヘッドの特徴量次元] ※QKV(3)×ヘッド数×ヘッドの特徴量次元 = dim_hidden*3
        #ここで、view() や reshape() を使う際には、元のテンソルの要素数と新しい形状のテンソルの要素数が一致している必要
        # があります。[-1]は、残りの次元を自動的に計算するための特殊な値です。この操作で次元数は変わらないのです！

        # permute関数により次元を入れ替えて、
        # [QKV（３）, バッチサイズ（bs）, ヘッド数(self.num_heads), 特徴量数(ns=num_patches + 1), ヘッドの特徴量次元(dim_head*3 = dim_hidden // num_heads)]とする
        qkv = qkv.view(
            bs, ns, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # unbind(0) を使って、qkv をクエリ (q)、キー (k)、バリュー (v) に分割します。
        # 各テンソルの次元は [バッチサイズ, ヘッド数, 特徴量数, ヘッドの特徴量次元] です。
        #unbind() 関数は、PyTorch の torch.Tensor に属するメソッドです.指定した次元でテンソルを分割します。
        q, k, v = qkv.unbind(0)

        #  今、q, k, vの次元は、[バッチサイズ, ヘッド数, 特徴量数, ヘッドの特徴量次元] となります。

        # クエリとキーの行列積とアテンションの計算(今回マスクは不使用) """QK^T"""
        # クエリとキーの内積を取ることで、クエリが特定のキーにどれだけ関連性があるか、つまりどれだけ「注意（アテンション）」を払うべきかを計算します。
        # attnは[バッチサイズ, ヘッド数, 特徴量数, 特徴量数]
        #（特徴量数,ヘッドの特徴量次元）×（ヘッドの特徴量次元,特徴量数）＝（特徴量数,特徴量数）
        attn = q.matmul(k.transpose(-2, -1))
        #k.transpose(-2, -1) は、キー (k) の最後の2つの次元を入れ替えます
        # （すなわち、[特徴量数, ヘッドの特徴量次元] を [ヘッドの特徴量次元, 特徴量数] にします）。
        #  次元を入れ替える（転置）することで、行列積は各次元（[]）に対する内積となる。例：matmul(A, A^T) の操作は、行列 A の行（または列）同士の内積を計算することになります。
        """
        高次元テンソルの行列積においても、基本的には「二つの行列を注目して、他の次元が一致するように組み合わせて計算する」という考え方を使っています。
       つまり、次元のうち計算の対象になる2つの次元を選び、他の次元を合わせて、二つの行列の積として考えるという方法です。これが、テンソル計算をシンプルに考えるための鍵です。
       ちなみに、matmulは入力のテンソルが2軸より大きな場合、最後の2軸を使って計算をします。それより手前の次元は独立して扱われ、次元毎の行列積を得られます。
       """





        attn = (attn * self.scale).softmax(dim=-1) #self.scale = (dim_head ** -0.5) : (√D/N_h)^-1
        #softmax(dim=-1):dim=-1 は最後の次元に沿ってソフトマックスを適用することを意味します。



        # アテンションとバリューの行列積によりバリューを収集
        
        #attnは[バッチサイズ, ヘッド数, 特徴量数, 特徴量数] × vは[バッチサイズ, ヘッド数, 特徴量数, ヘッドの特徴量次元]
        #[特徴量数,特徴量数]×[特徴量数,ヘッドの特徴量次元] ＝ [特徴量数,ヘッドの特徴量次元]
        #よって x=は[バッチサイズ, ヘッド数, 特徴量数, ヘッドの特徴量次元]
        x = attn.matmul(v)

        # permute関数により
        # [バッチサイズ, 特徴量数, ヘッド数, ヘッドの特徴量次元]
        # flatten関数により全てのヘッドから得られる特徴量を連結して、
        # [バッチサイズ, 特徴量数, ヘッド数 * ヘッドの特徴量次元]
        """flatten(dim) メソッドは、テンソルの指定された次元（dim）以降の次元をすべて1つに結合して、平坦化する（1次元にする）処理です。"""
        """各ヘッドがそれぞれ別の視点から特徴を抽出しているので、最終的にそれを一つにまとめる必要があります。
           この結合された特徴量は、次の全結合層（proj_out）でさらに処理され、入力として扱いやすい形になります。"""
        x = x.permute(0, 2, 1, 3).flatten(2)
        x = self.proj_out(x)   #self.proj_out = nn.Linear(dim_hidden, dim_hidden)

        return x


"""Transformerエンコーダ内のFNNの実装"""

class FNN(nn.Module):
    '''
    Transformerエンコーダ内の順伝播型ニューラルネットワーク
    dim_hidden     : 入力特徴量の次元
    dim_feedforward: 中間特徴量の次元
    '''
    def __init__(self, dim_hidden: int, dim_feedforward: int):
        super().__init__()

        self.linear1 = nn.Linear(dim_hidden, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_hidden)
        self.activation = nn.GELU()

        "nn.Linear は入力テンソルの最後の次元に対して線形変換を行い、その次元を指定された値に変更します。"

    '''
    順伝播関数
    x: 入力特徴量, [バッチサイズ, 特徴量数, 特徴量次元]
    '''
    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x


"""Transformerエンコーダ層の実装"""


class TransformerEncoderLayer(nn.Module):
    '''
    Transformerエンコーダ層
    dim_hidden     : 入力特徴量の次元
    num_heads      : ヘッド数
    dim_feedforward: 中間特徴量の次元
    '''
    def __init__(self, dim_hidden: int, num_heads: int,
                 dim_feedforward: int):
        super().__init__()

        self.attention = SelfAttention(dim_hidden, num_heads)
        self.fnn = FNN(dim_hidden, dim_feedforward)

        self.norm1 = nn.LayerNorm(dim_hidden)
        self.norm2 = nn.LayerNorm(dim_hidden)

    '''
    順伝播関数
    x: 入力特徴量, [バッチサイズ, 特徴量数（num_patches + 1）, 特徴量次元(num_hidden)]
    ここでの特徴量数とは、パッチ（位置埋め込み済）数 + クラス埋め込みの分を指します。
    '''
    def forward(self, x: torch.Tensor):
        x = self.norm1(x)
        x = self.attention(x) + x # [+x]はスキップコネクション（残差接続）をしてる。スキップコネクションは元の入力を出力に加えるため、勾配が直接伝わりやすくなり、勾配消失問題を軽減できます。
        x = self.norm2(x)
        x = self.fnn(x) + x  #fnnの中で一度次元が少し変わるが、最終的には元の次元に戻る

        return x  #x : [バッチサイズ, 特徴量数（num_patches + 1）, 特徴量次元(num_hidden)]



"""ここからが本番、Vision Transformerの実装です。"""

"""Vision Transformerの実装"""


class VisionTransformer(nn.Module):
    '''
    Vision Transformer
    num_classes    : 分類対象の物体クラス数
    img_size       : 入力画像の大きさ(幅と高さ等しいことを想定)
    patch_size     : パッチの大きさ(幅と高さ等しいことを想定)
    dim_hidden     : 入力特徴量の次元
    num_heads      : マルチヘッドアテンションのヘッド数
    dim_feedforward: FNNにおける中間特徴量の次元
    num_layers     : Transformerエンコーダの層数
    '''
    def __init__(self, num_classes: int, img_size: int,
                 patch_size: int, dim_hidden: int, num_heads: int,
                 dim_feedforward: int, num_layers: int):
        super().__init__()

        # 画像をパッチに分解するために、
        # 画像の大きさがパッチの大きさで割り切れるか確認
        assert img_size % patch_size == 0

        self.img_size = img_size
        self.patch_size = patch_size  #self.img_size と self.patch_size のように self を使うことで、クラスのインスタンスに属する変数として定義されます。
                                      #これにより、他のメソッド内でも self.img_size と self.patch_size を参照することができます。


        # パッチ数は縦・横ともにimg_size // patch_sizeであり、(正方形を仮定)
        # パッチ数はその2乗になる　　縦×横
        num_patches = (img_size // patch_size) ** 2

        # パッチ特徴量（dim_patch）はパッチを平坦化することにより生成されるため、
        # その次元はpatch_size * patch_size * 3 (RGBチャネル)
        dim_patch = 3 * patch_size ** 2




        
        """以下はパッチの特徴量を dim_patch から dim_hidden の次元に変換する全結合層です。
        この変換により、各パッチの特徴をTransformerが理解しやすい次元にすることが目的です。"""
        self.patch_embed = nn.Linear(dim_patch, dim_hidden)

        

        # 位置埋め込み(パッチ数 + クラス埋め込みの分を用意)
        #➡クラス埋め込みにも、位置埋め込みをするので、パッチ数 + 1の位置埋め込み次元となる。
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, dim_hidden))
        #サイズは [1, num_patches, dim_hidden] で、バッチサイズに合わせてブロードキャストされます。


        # クラス埋め込み
        self.class_token = nn.Parameter(
            torch.zeros((1, 1, dim_hidden)))
        #サイズは [1, 1, dim_hidden] で、全てのバッチに同じクラス埋め込みを追加するため、
        # バッチサイズ分だけ拡張(broadcast)されます。

        #最終的に、エンコーダ層に入る入力次元は、[バッチサイズ, num_patches + 1, dim_hidden] となります。
        

        # Transformerエンコーダ層
        """PyTorchの nn.ModuleList は、複数の層をリスト形式で保持する特殊なリストです。
           普通のPythonリストとは異なり、ModuleList はPyTorchの nn.Module の一部として機能し、
           リストに含まれるすべての層がモデルの一部と認識され、学習中に適切に動作するようになります。"""
        self.layers = nn.ModuleList([TransformerEncoderLayer(
            dim_hidden, num_heads, dim_feedforward
        ) for _ in range(num_layers)])

        # ロジットを生成する前のレイヤー正規化と全結合
        self.norm = nn.LayerNorm(dim_hidden)
        self.linear = nn.Linear(dim_hidden, num_classes)

    '''
    順伝播関数
    x           : 入力, [バッチサイズ, 入力チャネル数, 高さ, 幅]  :  まさにViTモデルに入る最初のデータです。
    return_embed: 特徴量を返すかロジットを返すかを選択する真偽値
    '''
    def forward(self, x: torch.Tensor, return_embed: bool=False):
        bs, c, h, w = x.shape

#最初に入力画像 x の形状を取り出し、それぞれの変数に格納しています。
# bs（バッチサイズ）、c（チャネル数）、h（高さ）、w（幅）です。



        # 入力画像の大きさがクラス生成時に指定したimg_sizeと
        # 合致しているか確認
        assert h == self.img_size and w == self.img_size


        # 高さ軸と幅軸をそれぞれパッチ数 * パッチの大きさに分解:
        # ここで、h // self.patch_size と w // self.patch_size がパッチの数（行数と列数）です。
        # [バッチサイズ, チャネル数, パッチの行数, パッチの大きさ,パッチの列数, パッチの大きさ]の形にする
        x = x.view(bs, c, h // self.patch_size, self.patch_size,w // self.patch_size, self.patch_size)

        # permute関数により
        # [バッチサイズ, パッチ行数, パッチ列数, チャネル, パッチの大きさ, パッチの大きさ]の形にする
        x = x.permute(0, 2, 4, 1, 3, 5)

        # パッチを平坦化
        # permute関数適用後にはメモリ上のデータ配置の整合性の関係で
        # view関数を使えないのでreshape関数を使用
        x = x.reshape(
            bs, (h // self.patch_size) * (w // self.patch_size), -1)
        #reshape() を使って、各パッチを1次元に平坦化します。
        # 最終的な形状は [バッチサイズ, パッチ数, dim_patch] となります。

        x = self.patch_embed(x) #パッチの特徴量を dim_patch から dim_hidden の次元に変換.
        #現在:[バッチサイズ, パッチ数, dim_hidden] 

        # クラス埋め込み (class_token) をバッチサイズに合わせて拡張し、各バッチに追加します。
        #引数 (bs, -1, -1) は、各次元のサイズを指定します。ここで -1 が意味するのは、
        # 「元のサイズを保持する」ということです。
        class_token = self.class_token.expand(bs, -1, -1)    

        """self.class_token = nn.Parameter(torch.zeros((1, 1, dim_hidden)))"""

        x = torch.cat((class_token, x), dim=1)
#torch.cat() を使って、クラス埋め込みとパッチ特徴量を結合します。
#最終的な次元は [バッチサイズ, num_patches + 1, dim_hidden] になります。
# +1 されているのは、クラス埋め込みが追加されたためです。
#torch.cat() は、2つ以上のテンソルを指定された次元で連結するための関数です。
#dim=1 は、テンソルをどの次元で連結するかを指定します。ここで dim=1 は**第1軸（パッチの数を示す次元）**でテンソルを連結することを意味します。

        #位置埋め込み
        """
        定義:self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim_hidden))
        """
#バッチ次元が 1 であれば、PyTorch の自動ブロードキャストが適用されて、pos_embed は x
#  と同じ形状である [バッチサイズ, num_patches + 1, dim_hidden] に仮想的に拡張されます。
#(自動ブロードキャスト)
        x += self.pos_embed    #[バッチサイズ, num_patches + 1, dim_hidden] 

        # Transformerエンコーダ層を適用
        """
        定義 : self.layers = nn.ModuleList([TransformerEncoderLayer(
            dim_hidden, num_heads, dim_feedforward ) for _ in range(num_layers)])  ちなみに: for _ in range(num_layers) は、指定された数だけエンコーダ層を作成するためのループです。
        """
        for layer in self.layers:
            x = layer(x)
        """
        #このループは、self.layers 内に含まれる すべての TransformerEncoderLayerのインスタンス
        # を順番に取り出しています。
        # self.layers には TransformerEncoderLayer のインスタンスが格納されているので、
        # 各要素 layer は TransformerEncoderLayer クラスのオブジェクトになります！！
        # 
        # 
        # x = layer(x):、PyTorch の nn.Module クラスの仕様により、この layer の forward() メソッド が呼び出されます。
        何故なら、layer は nn.Module を継承している、TransformerEncoderLayer クラスのオブジェクトであり、
        PyTorch では、nn.Module を継承したクラスは、__call__() メソッドが実装されており、このメソッドは
        forward() メソッドを呼び出すように実装されているため、layer(x) と書くと自動的に forward(x) が呼び出されます！
        """


        # クラス埋め込みをベースとしたの特徴量を抽出
        """
        [:,0]の、：はその次元に対してすべての要素を取ることを意味する。０はその次元の最初の要素を取ることを意味する。
        今,: はバッチサイズ次元,0 はnum_patches + 1の次元であるので、全てのバッチに対してnum_patches + 1の最初の次元、
        つまりクラス埋め込み（+1 の部分）を取得することになります。逆に、num_patchesはここで捨てられます。
        （セルフアテンションでクラス埋め込みトークンに情報を集約させたので、用済み。）
        """
        x = x[:, 0]   #クラス埋め込みトークン"のみ"を取り出します。 [バッチサイズ, num_patches + 1, dim_hidden]から[バッチサイズ, dim_hidden]に変換されます。
        #クラス埋め込みトークン = x(バッチサイズ, dim_hidden) となる。

        x = self.norm(x)  #Layer Normalization を適用しています。

        if return_embed:
            return x
        """
        return_embed は引数として渡されるブール値で、この値が True の場合は、
        現在の特徴量ベクトル（x）をそのまま返します.例えば、特徴抽出のみを目的とするような場面
        （他のタスクに使うため、画像から抽出した特徴量を取り出したい場合など）では、この x を直接返すことで、特徴量を活用できます!!
        ※デフォルトは False です。
        """
        x = self.linear(x)

        """
        self.linear は nn.Linear(dim_hidden, num_classes) として定義されています。
        この全結合層の役割は、クラス埋め込みトークンから得られた特徴量を用いて、最終的なクラスのロジットを計算することです。
        dim_hidden 次元の特徴ベクトル x を num_classes 次元に変換します。
        これは最終的に、モデルがどのクラスに入力画像が属するのかを予測するためのスコア（ロジット）を出力するためです。
        """

        return x

    '''
    モデルパラメータが保持されているデバイスを返す関数
    self.linear.weight.device は、linear レイヤー（最終的な全結合層）の重み（weight）が
    配置されているデバイスを取得します。
    '''
    def get_device(self):
        return self.linear.weight.device
    
    '''
    モデルを複製して返す関数
    copy.deepcopy(self) を使用して、現在のモデルインスタンスをディープコピーします。
    ディープコピーを行うと、元のモデルとは完全に独立した新しいインスタンスが作成されます。
    つまり、元のモデルのパラメータや構造を変更しても、このコピーされたモデルには影響しません（逆も同じ）。
    ディープコピーを使うことで、モデルの重みなども含めて完全に同じ状態のコピーが作られるため、オリジナルと同様に利用することができます。
    '''
    def copy(self):
        return copy.deepcopy(self)






"""学習・評価におけるハイパーパラメータやオプションの設定"""

class Config:
    '''
    ハイパーパラメータとオプションの設定
    '''
    def __init__(self):
        self.val_ratio = 0.2       # 検証に使う学習セット内のデータの割合
        self.patch_size = 4        # パッチサイズ
        self.dim_hidden = 512      # 隠れ層の次元
        self.num_heads = 8         # マルチヘッドアテンションのヘッド数
        self.dim_feedforward = 512 # Transformerエンコーダ層内のFNNにおける隠れ層の特徴量次元
        self.num_layers = 6        # Transformerエンコーダの層数
        self.num_epochs = 30       # 学習エポック数
        self.lr = 1e-2             # 学習率
        self.moving_avg = 20       # 移動平均で計算する損失と正確度の値の数
        self.batch_size = 32       # バッチサイズ
        self.num_workers = 2       # データローダに使うCPUプロセスの数
        self.device = 'cuda'       # 学習に使うデバイス
        self.num_samples = 200     # t-SNEでプロットするサンプル数


"""学習・評価を行う関数"""

def train_eval():
    config = Config()  #train_eval は、学習と評価をまとめて行う関数です。Config クラスのインスタンスを生成して、ハイパーパラメータを取得します。

    # 入力データ正規化のために学習セットのデータを使って
    # 各チャネルの平均と標準偏差を計算
    dataset = torchvision.datasets.CIFAR10(root='data', train=True, 
                                           download=True,transform=T.ToTensor())
    """
    T は、torchvision.transforms モジュールを T という短縮名でインポートしているものです。
    ToTensor() は、PIL イメージや NumPy の ndarray を PyTorch のテンソルに変換するための関数です。
    ToTensor() は画像のピクセル値を 0〜1 の範囲に正規化してくれる役割も果たします（もともと画像のピクセル値は 0〜255 の整数値です）。
    """
    
    channel_mean, channel_std = util.get_dataset_statistics(dataset)
    """
    get_dataset_statisticsは各チャネルの平均と標準偏差を計算する関数です。別のutil.pyファイルに定義されています。
    util.pyは別のファイルに定義しますのでご覧ください。
    """

    # 画像の整形を行うクラスのインスタンスを用意
    transforms = T.Compose((
        T.ToTensor(),
        T.Normalize(mean=channel_mean, std=channel_std),
    ))

    # 学習、評価セットの用意
    train_dataset = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True,
        transform=transforms)
    test_dataset = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True,
        transform=transforms)

    # 学習・検証セットへ分割するためのインデックス集合の生成
    val_set, train_set = util.generate_subset(
        train_dataset, config.val_ratio)

    print(f'学習セットのサンプル数　: {len(train_set)}')
    print(f'検証セットのサンプル数　: {len(val_set)}')
    print(f'テストセットのサンプル数: {len(test_dataset)}')

    # インデックス集合から無作為にインデックスをサンプルするサンプラー
    train_sampler = SubsetRandomSampler(train_set)

    # DataLoaderを生成
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, sampler=train_sampler)
    val_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, sampler=val_set)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers)

    # 目的関数の生成
    loss_func = F.cross_entropy

    # 検証セットの結果による最良モデルの保存用変数
    val_loss_best = float('inf')
    model_best = None

    # Vision Transformerモデルの生成
    model = VisionTransformer(
        len(train_dataset.classes), 32, config.patch_size,
        config.dim_hidden, config.num_heads, config.dim_feedforward,
        config.num_layers)

    # モデルを指定デバイスに転送(デフォルトはGPU)
    model.to(config.device)

    # 最適化器の生成
    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    for epoch in range(config.num_epochs):
        model.train()

        with tqdm(train_loader) as pbar:
            pbar.set_description(f'[エポック {epoch + 1}]')

            # 移動平均計算用
            losses = deque()
            accs = deque()
            for x, y in pbar:
                # データをモデルと同じデバイスに転送
                x = x.to(model.get_device())
                y = y.to(model.get_device())

                # パラメータの勾配をリセット
                optimizer.zero_grad()

                # 順伝播
                y_pred = model(x)

                # 学習データに対する損失と正確度を計算
                loss = loss_func(y_pred, y)
                accuracy = (y_pred.argmax(dim=1) == \
                            y).float().mean()

                # 誤差逆伝播
                loss.backward()

                # パラメータの更新
                optimizer.step()

                # 移動平均を計算して表示
                losses.append(loss.item())
                accs.append(accuracy.item())
                if len(losses) > config.moving_avg:
                    losses.popleft()
                    accs.popleft()
                pbar.set_postfix({
                    'loss': torch.Tensor(losses).mean().item(),
                    'accuracy': torch.Tensor(accs).mean().item()})

        # 検証セットを使って精度評価
        val_loss, val_accuracy = eval.evaluate(
            val_loader, model, loss_func)
        print(f'検証　: loss = {val_loss:.3f}, '
                f'accuracy = {val_accuracy:.3f}')

        # より良い検証結果が得られた場合、モデルを記録
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            model_best = model.copy()

    # テスト
    test_loss, test_accuracy = eval.evaluate(
        test_loader, model_best, loss_func)
    print(f'テスト: loss = {test_loss:.3f}, '
          f'accuracy = {test_accuracy:.3f}')

    # t-SNEを使って特徴量の分布をプロット
    util.plot_t_sne(test_loader, model_best, config.num_samples)




"""学習・評価の実行"""
train_eval()

