"""YOLOv8 検出モデル用の Grad-CAM。

AI が「どこを見て」齢期を判定したかをヒートマップで可視化する。
Detect ヘッドに入る3つのスケールの特徴マップ（ネックの出力）それぞれで
Grad-CAM を計算し、入力画像サイズにそろえて平均する。

ターゲット（逆伝播するスコア）は、NMS で残った検出の「判定クラスの確信度」。
1匹だけ選べばその個体の根拠、全部を選べば画像全体の根拠になる。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression


class YOLOGradCAM:
    def __init__(self, yolo, imgsz: int = 640, layers: list[int] | None = None):
        self.net = yolo.model.float().eval()
        self.names = yolo.names
        self.imgsz = imgsz
        detect = self.net.model[-1]
        # 既定は Detect ヘッドの入力層（YOLOv8s なら 15, 18, 21）
        self.layers = layers if layers is not None else list(detect.f)
        self.stride = int(max(self.net.stride))

    def _preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        lb = LetterBox((self.imgsz, self.imgsz), auto=False, stride=self.stride)
        img = lb(image=bgr)
        img = img[..., ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW（ultralytics の前処理と同じ）
        x = torch.from_numpy(np.ascontiguousarray(img)).float().div(255.0)[None]
        return x

    def __call__(self, bgr: np.ndarray, conf: float = 0.5, iou: float = 0.7, targets: list[int] | None = None):
        """Grad-CAM を計算する。

        Args:
            bgr: 元画像（BGR, HxWx3, uint8）
            conf: 検出の確信度しきい値
            targets: 根拠を見たい検出の番号（None なら全検出）

        Returns:
            cam: 元画像サイズの 0〜1 ヒートマップ（検出がなければ None）
            dets: 元画像座標の検出 [(x1, y1, x2, y2, conf, cls), ...]
        """
        acts, grads, handles = {}, {}, []
        for i in self.layers:
            def hook(_m, _inp, out, i=i):
                acts[i] = out
                out.register_hook(lambda g, i=i: grads.__setitem__(i, g))
            handles.append(self.net.model[i].register_forward_hook(hook))

        try:
            x = self._preprocess(bgr).requires_grad_(True)
            with torch.enable_grad():
                out = self.net(x)
                pred = out[0] if isinstance(out, (list, tuple)) else out  # (1, 4+nc, anchors)
                keep, idxs = non_max_suppression(pred.detach(), conf, iou, agnostic=True, return_idxs=True)
                keep, idxs = keep[0], idxs[0]
                if len(keep) == 0:
                    return None, []

                sel = range(len(keep)) if targets is None else [t for t in targets if 0 <= t < len(keep)]
                score = sum(pred[0, 4 + int(keep[j, 5]), int(idxs[j])] for j in sel)
                self.net.zero_grad(set_to_none=True)
                score.backward()
        finally:
            for h in handles:
                h.remove()

        h_in, w_in = x.shape[2:]
        cam = np.zeros((h_in, w_in), dtype=np.float32)
        for i in self.layers:
            a, g = acts[i].detach()[0], grads[i][0]
            w = g.mean(dim=(1, 2), keepdim=True)  # チャネルごとの重み = 勾配の空間平均
            c = torch.relu((w * a).sum(0)).numpy()
            c = cv2.resize(c, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
            if c.max() > 0:
                cam += c / c.max()  # スケールごとに正規化してから足す
        cam /= max(cam.max(), 1e-8)

        # レターボックスの余白を除き、元画像サイズに戻す
        h0, w0 = bgr.shape[:2]
        r = min(h_in / h0, w_in / w0)
        nh, nw = round(h0 * r), round(w0 * r)
        top, left = round((h_in - nh) / 2 - 0.1), round((w_in - nw) / 2 - 0.1)
        cam = cv2.resize(cam[top:top + nh, left:left + nw], (w0, h0), interpolation=cv2.INTER_LINEAR)

        boxes = ops.scale_boxes((h_in, w_in), keep[:, :4].clone(), (h0, w0))
        dets = [(*b.tolist(), float(k[4]), int(k[5])) for b, k in zip(boxes, keep)]
        return cam, dets


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """RGB 画像にヒートマップ（赤ほど判定に効いた場所）を重ねる。"""
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[..., ::-1]
    return np.uint8(rgb * (1 - alpha) + heat * alpha)


if __name__ == "__main__":
    import argparse

    from PIL import Image
    from ultralytics import YOLO

    p = argparse.ArgumentParser(description="YOLOv8 Grad-CAM（アブラムシ齢期判定）")
    p.add_argument("image")
    p.add_argument("--weights", default="best.pt")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--out", default="gradcam.jpg")
    a = p.parse_args()

    rgb = np.array(Image.open(a.image).convert("RGB"))
    gc = YOLOGradCAM(YOLO(a.weights))
    cam, dets = gc(np.ascontiguousarray(rgb[..., ::-1]), conf=a.conf)
    if cam is None:
        raise SystemExit("検出なし")
    vis = overlay(rgb, cam)
    for x1, y1, x2, y2, c, k in dets:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), max(2, rgb.shape[1] // 500))
        print(f"{gc.names[k]}  conf={c:.2f}  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    Image.fromarray(vis).save(a.out, quality=92)
    print("saved", a.out)
