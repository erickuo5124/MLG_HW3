# HW3: Model Design & Comparison for Recommendation Models
###### tags: `MLG`

## 🌏 環境

```shell
$ uname -a
Darwin uuMacBook.local 20.4.0 Darwin Kernel Version 20.4.0: Thu Apr 22 21:46:41 PDT 2021; root:xnu-7195.101.2~1/RELEASE_ARM64_T8101 x86_64
```

### 套件使用

- python 3.7.0
- torch 1.7.0
- torch-geometric 1.7.0

因為我的筆電是 ARM 架構的，沒辦法使用 pytorch ，但又很想試試在比較有架構的方式下開發，因此稍微查了一下如何在這個情況下安裝 pytorch

### 安裝步驟

1. 下載 [Miniforge3](https://github.com/conda-forge/miniforge#download)

選擇目前作業系統下的去下載，完成之後執行檔案

```shell
bash Miniforge3-Linux-x86_64.sh
```

2. 創建 conda 的環境，並進入該環境

```shell
conda create -n mlg
conda activate mlg
```

:::info
可以使用 `conda env list` 查看現有環境列表
:::

3. 安裝合適的 python 版本

```shell
conda install python=3.7
```

:::warning
pytorch 目前不支援 3.8 以上
:::

4. 安裝套件

```shell
conda install pytorch
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cpu.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cpu.html
pip install torch-geometric
```

也可以到 [下載點](https://pytorch-geometric.com/whl/) 選擇合適的版本下載，因為 M1 似乎沒辦法用 cuda ，所以選擇 cpu 版本下載。

---

## 📦 資料

### [MovieLens](https://drive.google.com/file/d/1xkBBoKVI_Ksd4plkZKbTMrgD54AGFBRo/view?usp=sharing)

關於電影評分的資料集

| Entity | #Entity | 
| -------- | -------- |
| user | 943 |
| movie | 1682 |
| age (user) | 8 |
| occupation (user) | 21 |
| genre (movie) | 18 |

| Relation | #Relation |
|-|-|
|user-movie|100k|
|user-age|943|
|user-occupation|943|
|movie-genre| 2861 |
|user-user|47150|
|movie-movie|82798|

### [Yelp](https://drive.google.com/file/d/1CnSUKXZJKPmHUdmm0eepVM1vXk2up6kS/view?usp=sharing)

一個用戶對餐館等場所進行評價的網站

| Entity | #Entity | 
|-|-|
|user|16239|
|business|14284|
|compliment (user)|11|
|category (business)|511|
|city (business) | 47 |

| Relation | #Relation |
|-|-|
|user-business|198397|
|user-user|158590|
|user-compliment|76875|
|business-city|14267|
|business-category|40009|

### [Douban Book](https://drive.google.com/file/d/1uTCUZ9Me1MlvQIkb5oeGG2yqINPCFsWK/view?usp=sharing)

豆瓣圖書，有許多使用者對書本的評價

| Entity | #Entity | 
|-|-|
|user|13024|
|book|22347|
|group (user)|2936|
|location (user)|38|
|author (book)|10805|
|publisher (book)|1815|
|year (book)|64|

|Relation|#Relation|
|-|-|
|user-book|792062|
|user-group|1189271|
|user-user|169150|
|user-location|10592|
|book-author|21907|
|book-publisher|21773|
|book-year|21192|

### 資料處理

1. 讀入資料，資料大概分成兩種形式：
    (1) a, b, rating 
    (2) a, b
    ，將矩陣初始化成 0 之後，第一種將 rating 填入，第二種則是用 0/1 的形式輸入
2. 將 interaction < 3 的資料篩選掉
3. 存成 .pkl 檔，以便之後使用

---

## ❓ Problem Statement


![](https://i.imgur.com/rPXMYpv.png)

- Input: user 與 item 的互動資料，以及 user 跟 item 的 feature
- Output: user 對 item 的 rating


---

## 📚 Typical RecSys Methods

### CF (Collaborative Filtering)

利用資料特徵的相似度來做推薦

#### User-based

比對 user 的 feature ，找到前幾相似的 user ，並利用那些 user 的rating 來預測。

#### Item-based

原理同 User-based ，實作上將矩陣行列互換就好了。

#### similarity

- cosine similarity

$$
sim(u, v) = cos(r_u, r_v) = \frac{r_u \cdot r_v}{||r_u||\cdot ||r_v||}
$$

- pearson correlation coefficient

$$
sim(u, v) = \frac{
    \sum_{i\in S_{uv}}(r_{ui}-\bar r_u)(r_{vi}-\bar r_v)
}{
    \sqrt{\sum_{i\in S_{uv}}(r_{ui}-\bar r_u)^2}
    \sqrt{\sum_{i\in S_{uv}}(r_{vi}-\bar r_v)^2}
}
$$

#### predicted rating

將要預測的 item $i$ 的所有評分加起來做平均

$$
r_{ui} = \frac{1}{k}\sum_{v\in N}{r_{vi}}
$$

或（但我一直寫不出來這個QAQ）

$$
r_{ui} = \frac{\sum_{v\in N}{s_{uv}r_{vi}}}{\sum_{v\in N}s_{uv}}
$$

### MF (Matrix Factorization)

以矩陣分解並重組成 user 對 item 的矩陣

![](https://i.imgur.com/oUPEbPX.png)

$$
Y = X \Theta^T
$$

- $Y$：user 對 item 的矩陣
- $X$：user feature
- $\Theta$：item feature
- k：自訂的 feature 維度

#### 實作步驟

1. 隨機生成 $X$ 跟 $\Theta$
2. 學習 $X$ 的參數
3. 學習 $Theta$ 的參數
4. 重複 2. 跟 3. ，直到可接受的 loss 值

但是預測的區域必須假設沒有資料，因此上述的步驟得執行兩次，分別學習 user feature 跟 item feature 

![](https://i.imgur.com/RYQiUEQ.png)

![](https://i.imgur.com/LYuKHia.png)

如此才能在不看到要預測的資料的情況下，同時學習到完整的 user feature 跟 item feature

---

## 🌐 NN‐based RecSys Methods

---

## 💻 Recent NN‐based Methods

---

## 📈 Evaluation

### 5‐Fold Cross Evaluation

![](https://i.imgur.com/0vql5h2.png)

資料會被分成訓練跟驗證兩部分：

- 訓練（80%）：訓練模型時只會看到這部分，而這裡又被細分為兩部分
    - training（70%）：訓練時使用的資料
    - validation（10%）：訓練時沒看過的資料，用於即時判斷模型效果
- 驗證（20%）：最後評估模型好壞時使用的資料

### RMSE

Root-Mean-Square-Error ，我使用了 sklearn 的 MSE 加上平方根

$$
RMSE(ans, pred) = \sqrt{\sum{(ans-pred)^2}}
$$

### Recall

將推薦出來的物品依照相關性來做評分，相關性的值為 0/1 ，但因為資料並非完全是以 0/1 作為評分，因此我將資料做以下的轉換：

1. 取得平均的評分值 $avg$
2. r > $avg$：1,  r < $ang$：0

而 recall 的公式如下：

$$
Recall = \frac{|relevant\bigcap retrived|}{|relevant|}
$$

retrived 即為 predict 中推薦的物品，而 relevant 則是實際上使用者喜歡的物品

將資料做轉換之後，可以得到以下的表格

||retrived|not retrived|
|-|-|-|
|relevant|True Positive (TP)|False Negative (FN)|
|irrelevant|False Positive (FP)|Ture Negative(TN)|

因此也可以這樣看 recall 的公式：

$$
Recall = \frac{|TP|}{|TP|+|TN|}
$$

### NDCG

- G (Gain)：一個物品的相關性，這裡用實際的評分資料來代表
- CG (Cumulative Gain)：累積的相關性，可以對推薦列表做評分，公式為

$$
CG = \sum G_i
$$

- DCG (Discount Cumulative Gain)：考慮到排序的先後順序，並依照順序對值做折扣，公式為

$$
DCG = \sum \frac{G_i}{log_2i}
$$

log 的底數取越大，折扣的效果越大，我使用 2 來當作底數

- IDCG（Ideal Discount Cumulative Gain）：理想狀態中的物品推薦排序，應該會是資料集的最大值
- NDCG（Normalize Discount Cumulative Gain）：對 DCG 做正規化，公式為

$$
NDCG = \frac{DCG}{IDCG}
$$

實作上並沒有預測排序，因此我把預測的 rating 當作排序的依據，也就是 rating 越大排序越前面，至於相關性 $G_i$ 則是利用 rating 的實際值，因此 IDCG 就是將實際上的 rating 做由大到小的排序，並將 rating 看成是 $G_i$ 帶入 DCG 的公式。正常來說 IDCG 應該要大於 DCG

---

## 參考資料

- [文件檢索的評價](https://ithelp.ithome.com.tw/articles/10192869)