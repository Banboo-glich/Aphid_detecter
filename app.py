import os
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

st.set_page_config(page_title="AphidDetector", layout="wide")

MODEL_PATH = "best.pt"
ASSETS = "assets"

# モデルのクラス番号 → 日本語の名前（5th は成虫）
JP_NAMES = {0: "1齢", 1: "2齢", 2: "3齢", 3: "4齢", 4: "成虫"}
COLORS = {0: (230, 60, 60), 1: (240, 150, 40), 2: (60, 170, 60), 3: (40, 120, 230), 4: (150, 60, 200)}

# 論文 Fig.3B の数値（検証データ 322枚）
PAPER_METRICS = pd.DataFrame({
    "齢期": ["1齢", "2齢", "3齢", "4齢", "成虫"],
    "適合率": [0.969, 0.828, 0.892, 0.914, 0.920],
    "再現率": [0.887, 0.947, 0.886, 0.903, 0.970],
    "mAP50": [0.980, 0.951, 0.959, 0.980, 0.991],
})
# 論文 Fig.3C 混同行列の対角（本物のうち正しく答えた割合）
PAPER_CORRECT = pd.DataFrame({
    "齢期": ["1齢", "2齢", "3齢", "4齢", "成虫"],
    "正しく答えた割合(%)": [87, 93, 88, 92, 97],
})

PAPER_CITE = ("出典：Masuko M, Kikuta S (2024) Image recognition-based deep learning model for identifying "
              "the developmental stages of *Acyrthosiphon pisum*. Applied Entomology and Zoology. "
              "https://doi.org/10.1007/s13355-024-00873-w")


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


@st.cache_resource
def load_font(size=28):
    for path in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return None  # 日本語フォントがないときは英語ラベルにする


def detect(image, conf):
    """画像を判定して、枠を描いた画像と齢期ごとの数を返す"""
    model = load_model()
    rgb = np.array(image.convert("RGB"))
    result = model.predict(np.ascontiguousarray(rgb[..., ::-1]), conf=conf, agnostic_nms=True, verbose=False)[0]  # 学習時と同じBGRで渡す

    out = Image.fromarray(rgb)
    draw = ImageDraw.Draw(out)
    font = load_font(max(20, out.width // 60))
    line = max(3, out.width // 400)
    counts = Counter()
    for (x1, y1, x2, y2), c, p in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                      result.boxes.conf.tolist()):
        c = int(c)
        counts[JP_NAMES[c]] += 1
        label = f"{JP_NAMES[c]} {p:.2f}" if font else f"{model.names[c]} {p:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=COLORS[c], width=line)
        tw, th = (draw.textlength(label, font=font), font.size + 8) if font else (len(label) * 7, 14)
        ty = y1 - th if y1 >= th else y1
        draw.rectangle([x1, ty, x1 + tw + 10, ty + th], fill=COLORS[c])
        draw.text((x1 + 5, ty + 2), label, fill="white", font=font)
    return out, counts


def count_table(counts):
    return pd.DataFrame({"齢期": list(JP_NAMES.values()),
                         "匹数": [counts.get(n, 0) for n in JP_NAMES.values()]})


def page_intro():
    st.title("エンドウヒゲナガアブラムシの齢期をAIで判定する")
    st.write("実体顕微鏡で撮ったアブラムシの写真から、AI（YOLOv8）が1匹ずつ"
             "「1齢・2齢・3齢・4齢・成虫」のどれかを判定します。")

    st.header("なぜAIで見分けるのか")
    st.write("アブラムシは体長が数mmと小さく、齢期ごとの見た目の差もわずかです。"
             "これまでは1匹ずつ飼って脱皮殻の数を数えるなど、人の目と手間に頼っていました。"
             "写真から一瞬で齢期が分かれば、発育の研究や、畑での害虫の調査を速く正確に進められます。")

    st.header("研究のようす")
    c1, c2 = st.columns(2)
    c1.image(os.path.join(ASSETS, "lab_setup.jpg"), width="stretch",
             caption="撮影のようす：実体顕微鏡にカメラをつなぎ、パソコンで撮影します")
    c2.image(os.path.join(ASSETS, "lab_rearing.jpg"), width="stretch",
             caption="飼育のようす：ソラマメの葉を入れた小さな容器で、1匹ずつ育てて齢期を記録します")

    st.header("写真で見る（解析の前と後）")
    st.write("1枚の写真に、成虫と幼虫が一緒に写っています。右がAIの判定結果です。")
    img = Image.open(os.path.join(ASSETS, "sample_multi.jpg"))
    out, counts = detect(img, 0.5)
    c1, c2 = st.columns(2)
    c1.image(img, caption="解析前", width="stretch")
    c2.image(out, caption="解析後（枠の数字はAIの確信度）", width="stretch")
    st.table(count_table(counts).set_index("齢期").T)

    st.header("動画で見る（解析の前と後）")
    st.write("脱皮の動画は、AIが1コマずつ判定しています（左が元の動画、右が判定結果）。")
    st.subheader("脱皮")
    st.write("古い皮（白い部分）から体を抜け出しているところです。")
    st.video(os.path.join(ASSETS, "molting.mp4"), loop=True, autoplay=True, muted=True)
    st.subheader("大きいアブラムシをじっくり見て下さい。出産してます。")
    st.write("アブラムシの成虫は、卵ではなく子どもを産みます（胎生）。生まれた子どもが1齢幼虫です。")
    st.video(os.path.join(ASSETS, "childbirth.mp4"), loop=True, autoplay=True, muted=True)


def page_accuracy():
    st.title("AIはどれくらい正確か")
    st.write("学習に使っていない写真322枚（検証データ）でテストした結果です。数値はすべて論文から引用しています。")

    c1, c2, c3 = st.columns(3)
    c1.metric("見分ける段階", "5段階")
    c2.metric("総合点 mAP50", "すべて95%以上")
    c3.metric("F1スコア（全体）", "0.91")

    st.header("本物100匹のうち、何匹を正しく答えたか")
    st.bar_chart(PAPER_CORRECT.set_index("齢期"))
    st.write("どの齢期でも、およそ9割前後を正しく見分けています。"
             "間違えるときは、となりの齢期（例：3齢を2齢）と間違えることがほとんどでした。"
             "体の大きさが近い齢期どうしは、人が見ても迷いやすいところです。")

    st.header("くわしい数値")
    st.bar_chart(PAPER_METRICS.set_index("齢期"), stack=False)
    st.dataframe(PAPER_METRICS, hide_index=True)
    st.markdown(
        "- **適合率**：AIが「○齢」と答えたもののうち、本当に○齢だった割合\n"
        "- **再現率**：本物の○齢のうち、AIが見つけられた割合\n"
        "- **mAP50**：見つける力と見分ける力をまとめた総合点（物体検出でよく使われる指標）\n"
        "- **F1スコア**：適合率と再現率のバランスを見る指標")

    st.header("いろいろな条件でも判定できる")
    st.image(os.path.join(ASSETS, "paper_fig4.jpg"), width="stretch")
    st.markdown(
        "- **A**：自然光・実体顕微鏡（4齢、確信度0.92）\n"
        "- **B**：成虫と生まれたばかりの1齢幼虫を同時に判定\n"
        "- **C**：学習に使っていない別の顕微鏡・カメラ（4齢、確信度0.95）\n"
        "- **D**：スマートフォン＋拡大レンズ（4齢、確信度0.96）")

    st.image(os.path.join(ASSETS, "paper_fig5.jpg"), width="stretch")
    st.write("上から自然光・白色光・黄色光、左から1齢・2齢・3齢・4齢・成虫。光の色が変わっても判定できるように、"
             "条件を変えて撮った写真で学習させています。")

    st.header("AIができるまで")
    st.markdown(
        "1. **撮影**：実体顕微鏡（15〜45倍）で、光の種類を変えながら撮影\n"
        "2. **印つけ（アノテーション）**：写真の虫を四角で囲み、齢期の名前をつける（脱皮殻で齢期を確認）\n"
        "3. **学習**：学習用1,235枚を、YOLOv8で300回くり返し学習\n"
        "4. **評価**：検証用322枚でテストし、5種類のモデルから最も良いYOLOv8sを選択\n"
        "5. **公開**：誰でも使えるようにWebアプリとして公開")

    st.header("これからの課題")
    st.write("顕微鏡のピントが合っていない写真では、正しく判定できる割合が下がります。"
             "また、羽が生える前の幼虫（翅芽の有無）を見分けることはまだできません。")

    st.caption(PAPER_CITE)
    st.caption("データセット：https://doi.org/10.5281/zenodo.10811254　ソースコード：https://github.com/Banboo-glich/Aphid_detecter")


def page_detect():
    st.title("判定してみる")
    st.write("実体顕微鏡で撮った写真をアップロードすると、AIが齢期を判定して数えます。")

    st.sidebar.markdown("----")
    conf = st.sidebar.slider("確信度のしきい値", 0.0, 1.0, 0.50,
                             help="この値より自信のない判定は表示しません。見落としが多いときは下げてください。")
    file = st.sidebar.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

    img = Image.open(file) if file else Image.open(os.path.join(ASSETS, "sample_multi.jpg"))
    if not file:
        st.info("画像がないときはサンプル画像で判定します。")

    out, counts = detect(img, conf)
    c1, c2 = st.columns(2)
    c1.image(img, caption="解析前", width="stretch")
    c2.image(out, caption="解析後", width="stretch")

    if sum(counts.values()) == 0:
        st.warning("アブラムシが見つかりませんでした。しきい値を下げるか、別の画像を試してください。")
    else:
        st.write(f"検出数：{sum(counts.values())}匹")
        st.table(count_table(counts).set_index("齢期").T)


def main():
    st.sidebar.title("Settings")
    mode = st.sidebar.selectbox("ページを選ぶ", ["はじめに", "AIの精度", "判定してみる"])
    if mode == "はじめに":
        page_intro()
    elif mode == "AIの精度":
        page_accuracy()
    else:
        page_detect()


if __name__ == "__main__":
    main()
