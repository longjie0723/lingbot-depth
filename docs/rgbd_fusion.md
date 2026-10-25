# RGB-D 融合（ブランチ `rgbd-fusion`）

LingBot-Depth は計測深度をエンコーダの入力に使いますが、デコーダが見るのは RGB のトークンだけです。そのため出力は計測値に大まかにしか従いません。公開版のモデルは、計測できている画素でもセンサー自体より不正確です。たとえば RobbyVla（D405）では、誤差の中央値がモデル 6.9 mm に対し、センサーは 1.0 mm です。

このブランチでは、モデルの中に学習による融合の仕組みを加えます。計測が信頼できる画素では計測値を保ち、穴は計測値とつじつまが合うように埋めます。処理時間の増加はわずかです。

研究用途に限ります。融合部分の重みは [robbyant/mdm_depth](https://huggingface.co/datasets/robbyant/mdm_depth)（CC BY-NC-SA 4.0）で学習しているので、商用には使えません。

## モデルの変更

以下の 2 つ（`depth_reinject` と `refiner`）は `MDMModel` のオプションで、設定はチェックポイントに保存されます。既定ではオフなので、公開チェックポイントはこれまでどおり読み込めて、出力も変わりません。

- **`depth_reinject=True`:** デコーダ（neck）のレベル 1〜4 に、計測深度の対数とその被覆率を追加で入力します。どちらも各レベルの解像度に縮小したものです。
  - 追加分の重みはゼロ初期化です（`MDMModel.adapt_state_dict`）。
  - そのため公開チェックポイントは、学習するまで出力がまったく変わりません。
- **`refiner={...}`:** `AnchoredRefiner`（`mdm/model/modules_refiner.py`、約 5.1 万パラメータ）が、デコーダの出力をフル解像度で補正します。補正後は、補正前の深度 `depth_base` と信頼度 `confidence` も返します。
  1. 計測値をどれだけ信頼するかを、画素ごとに推定します。
  2. 信頼できる画素は、計測値に合わせます。
  3. 信頼できる画素での計測値と予測のずれ log(raw) - log(pred) を、学習した近傍の重みで、粗い解像度から細かい解像度へ順に周囲へ広げます。こうして穴は、予測した形を保ったまま、周囲のずれを受け継ぎます。
  4. 穴がずれをどこまで受け継ぐかは、次のどちらかで決めます。
     - `gate='scalar'`：画像全体で 1 つの学習済みの値
     - `gate='pixel'`：画素ごとの値。特徴量と、周囲にある信頼できる計測点の密度から予測します。
- **`infer(..., amp_dtype=torch.float16)`:** 自動混合精度の既定は bf16 ですが、Ampere より前の GPU では xformers のアテンションが bf16 に対応していません。fp16 にしても、fp32 との差は相対で約 2e-4 です。

## データ

`tools/stream_subset.py` は、データセットのアーカイブ（50〜320 GB）をダウンロードせず、ストリーミングで読みながら必要な部分だけを取り出します（`curl | zstd | tar`）。取り出したデータは `mdm/data/mdm_depth.py` で読み込みます。

- **評価用:**

  | サブセット | 内容 | 枚数 | 取り出し方 |
  |---|---|---|---|
  | RobbySimVal | シミュレーション | 1557 | `--frame-stride 25` |
  | RobbyReal | RealSense D415/D435/D455、部屋 00001_01〜04 | 84 | `--frame-stride 100` |
  | RobbyVla | D405 | 627 | 10 フレームごと |

- **学習用:** 元の解像度のまま 12.5 万枚、132 GB。評価用とシーンが重ならないようにしています。

  | サブセット | 内容 | 枚数 | 取り出し方 |
  |---|---|---|---|
  | RobbySim | カメラグループ 6 つごとに 1 つ | 5.5 万 | `--group-stride 6` |
  | RobbyReal | 評価用の部屋を除くアクティブステレオカメラ | 2.7 万 | `--include '^RobbyReal/(?!00001_0[1-4]_)' --frame-stride 10` |
  | RobbyVla | 評価用のグループを除く | 4.4 万 | `--frame-stride 20` |

## ツール

| ツール | 用途 |
|---|---|
| `tools/eval_depth.py` | 計測画素・穴・全画素での真値との誤差と、計測深度との差を計算する（疑似的な穴での評価もできる） |
| `tools/compare_eval.py` | 複数の `eval_depth.py` の結果を表にして並べる |
| `tools/analyze_fusion_headroom.py` | 学習なしの基準を計算する（画素ごとの単純な切り替え規則、理論上の上限） |
| `tools/bench_latency.py` | 処理時間を段階ごとの内訳つきで測る |
| `tools/train_fusion.py` | エンコーダを固定して融合部分を学習する（DDP、fp16、解像度ごとにまとめたバッチ） |

```bash
# 学習（GPU 2 台）: refiner + neck + depth head、画素ごとのゲート
torchrun --nproc_per_node 2 tools/train_fusion.py --train-root data/mdm_depth_train --val-root data/mdm_depth \
    --model-kwargs '{"depth_reinject": true, "refiner": {"gate": "pixel"}}' \
    --trainable refiner neck depth_head --w-base 0.5 --iters 20000 --out runs/full_pixgate

# 評価（SimVal と Real/Vla は別々に実行する。疑似的な穴の位置をモデル間でそろえるため）
python tools/eval_depth.py --data-root data/mdm_depth --subsets RobbySimVal --model runs/full_pixgate/latest.pt \
    --tokens 1200 --pseudo-hole-ratio 0.15 --out results/full_pixgate/simval
python tools/eval_depth.py --data-root data/mdm_depth --subsets RobbyReal RobbyVla --model runs/full_pixgate/latest.pt \
    --tokens 1200 --pseudo-hole-ratio 0.15 --out results/full_pixgate/real
python tools/compare_eval.py base=results/base/simval,results/base/real full_pixgate=results/full_pixgate
```

## 結果

**条件**
- データ：評価データ全体
- 推論：1200 トークン（resolution level 0）、fp16
- モデル：2 万回学習した最終チェックポイント（GPU 2 台 × バッチ 4、エンコーダ固定）

**表の見方**
- 数値はサンプルごとの中央値です。
- 「計測」と「穴」は誤差の中央値（MedAE、mm）です。「計測」は計測値がある画素、「穴」は計測値がない画素です。
- 「全体」は、真値がある全画素での相対誤差（AbsRel）です。

| モデル | Real 計測 | Real 穴 | Real 全体 | Vla 計測 | Vla 穴 | Vla 全体 | SimVal 計測 | SimVal 穴 | SimVal 全体 |
|---|---|---|---|---|---|---|---|---|---|
| 計測深度 | 4.0 | - | - | 1.0 | - | - | 4.0 | - | - |
| 公開版モデル | 5.5 | 11.2 | 0.0108 | 6.9 | 10.1 | 0.1434 | 13.6 | 19.4 | 0.0160 |
| gate@0.05（学習なし） | 4.0 | 11.2 | 0.0123 | 2.0 | 10.1 | 0.1366 | 6.0 | 19.4 | 0.0124 |
| 計測画素での理論上の上限 | 2.3 | 11.2 | 0.0068 | 1.0 | 10.1 | 0.1097 | 3.3 | 19.4 | 0.0094 |
| refiner、1 つの値のゲート | 3.3 | 10.1 | 0.0107 | 1.4 | 7.8 | 0.0901 | 8.1 | 17.3 | 0.0134 |
| refiner + neck + head、1 つの値のゲート | 3.7 | 9.9 | 0.0092 | 1.1 | 4.6 | 0.0537 | 9.0 | 18.4 | 0.0134 |
| refiner、画素ごとのゲート | 3.3 | 11.3 | 0.0095 | 1.0 | 3.7 | 0.0665 | 9.7 | 19.4 | 0.0146 |
| **refiner + neck + head、画素ごとのゲート** | 3.7 | **9.6** | **0.0090** | **1.0** | **2.4** | **0.0386** | 9.2 | 18.2 | 0.0137 |

学習なしの 2 つの基準は、どちらも計測画素だけを変えます（`tools/analyze_fusion_headroom.py`）。
- **gate@0.05:** 計測値が予測の 5% 以内に収まる画素では計測値を使い、それ以外は予測を使います。
- **理論上の上限:** 計測値と予測のうち、真値に近いほうを選んだ場合です。

**わかったこと**
- **RealSense のデータ（Real：D415/D435/D455、Vla：D405）:**
  - 計測値を保てています。Vla はセンサー自体と同じ 1.0 mm で、Real は 76% のフレームで計測深度より正確です。
  - 穴は公開版モデルよりずっと正確に埋まります。Vla の穴の誤差は 10.1 mm から 2.4 mm になりました。
- **画素ごとのゲートの効果:** 出力に計測パターンの跡が残る問題がなくなりました。1 つの値のゲートでは、計測点のかたまりの間にある穴に補正が一部しか届かず、まばらな計測の模様が出力に写っていました。
- **RobbySimVal:** 学習なしの単純な切り替え（gate@0.05）にも届いていません。
  - 原因は、学習用の RobbySim シーンに付いている唯一の計測深度（`*_rmd2c.png`）が、RobbySimVal の計測深度よりかなりノイジーなことです。真値との差が 1% 以内の画素は、学習用が 56%、評価用が 81% です。
  - その結果、モデルはシミュレーションの計測値を信用しないことを学習しています。計測画素での信頼度の平均は、SimVal で約 0.5、実データで約 0.8 です。
  - 計測値をどれだけ信用するかは、学習に使ったセンサーから学ばれます。製品向けのモデルは、そのカメラ自身のデータで学習するべきです。

RTX 2080 Ti（fp16、`tools/bench_latency.py`）での処理時間:

| 入力解像度 | 公開版モデル | refiner + neck + head、画素ごとのゲート |
|---|---|---|
| 1280x960 | 105 ms | 122 ms |
| 1280x1024 | 109 ms | 127 ms |
