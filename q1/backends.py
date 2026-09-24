"""Lazy adapters for the four pinned model families and openSMILE.

Only the selected stage loads its models. No generated labels are used as truth.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from .common import atomic_json, atomic_npz, read_json, snapshot
from .temporal import conv_geometry, conv_intervals, pool_tokens, sampled_support, word_spans


def hf_model(spec, auto_class, device):
    import torch
    dtype = getattr(torch, spec["dtype"])
    if device == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    if dtype == torch.bfloat16 and device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise ValueError("此GPU不支持bfloat16；V100请用float16或float32")
    return auto_class.from_pretrained(spec["id"], revision=spec["revision"], dtype=dtype,
        attn_implementation=spec["attention"]).to(device).eval()


class Aligner:
    def __init__(self, cfg):
        from transformers import AutoModelForTokenClassification, AutoProcessor
        self.spec = cfg["models"]["align"]
        self.processor = AutoProcessor.from_pretrained(self.spec["id"], revision=self.spec["revision"])
        if not hasattr(self.processor, "prepare_forced_aligner_inputs"):
            raise RuntimeError("请安装支持原生Qwen3强制对齐的Transformers；本项目验证版本见requirements")
        self.model = hf_model(self.spec, AutoModelForTokenClassification, cfg["device"])

    def process(self, sample, directory, cfg):
        import torch
        media = read_json(directory / "media.json")
        wave, sr = sf.read(directory / "audio.wav", dtype="float32")
        if sr != 16000:
            raise ValueError("对齐输入必须是16 kHz")
        inputs, word_lists = self.processor.prepare_forced_aligner_inputs(
            audio=wave, transcript=sample["raw_text"], language="English", return_tensors="pt")
        inputs = inputs.to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            out = self.model(**inputs)
        decoded = self.processor.decode_forced_alignment(out.logits, inputs["input_ids"], word_lists,
            timestamp_token_id=self.model.config.timestamp_token_id)[0]
        if not decoded:
            raise ValueError("强制对齐未返回任何词")
        spans = word_spans(sample["raw_text"], [w["text"] for w in decoded])
        words, previous = [], 0.0
        for item, span in zip(decoded, spans):
            start, end = float(item["start_time"]), float(item["end_time"])
            if not np.isfinite([start, end]).all():
                raise ValueError("强制对齐产生非有限时间戳")
            flags = []
            if end <= start:
                flags.append("non_positive_duration")
            if start < previous - 1e-6:
                flags.append("overlap_or_nonmonotonic")
            if start < 0 or end > media["audio_duration"] + 1e-3:
                flags.append("out_of_audio_bounds")
            if media["silent_audio"]:
                # The aligner still returns times on digital silence; they carry no evidence.
                flags.append("silent_audio")
            previous = max(previous, end)
            # Retain raw timestamps. Clipping only bounds stored intervals; validity remains false.
            safe = np.clip([start, end], 0, media["audio_duration"]) + media["audio_offset"]
            words.append({"text": item["text"], "char_span": span,
                "raw_audio_interval": [start, end], "start": float(safe[0]), "end": float(safe[1]),
                "time_valid": not flags, "flags": flags})
        result = {"words": words, "model": self.spec, "time_unit": "seconds",
                  "time_reference": "clip origin defined in media.json",
                  "confidence": None, "note": "无校准置信度；仅记录结构性时间检查，不宣称对齐准确率"}
        atomic_json(directory / "align.json", result)
        return {"words": len(words), "valid_words": sum(w["time_valid"] for w in words)}


class TextEncoder:
    def __init__(self, cfg):
        from transformers import AutoModel, AutoTokenizer
        self.spec = cfg["models"]["text"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec["id"], revision=self.spec["revision"], use_fast=True)
        self.model = hf_model(self.spec, AutoModel, cfg["device"])

    def process(self, sample, directory, cfg):
        import torch
        words = read_json(directory / "align.json")["words"]
        tokens = self.tokenizer(sample["raw_text"], return_offsets_mapping=True,
                                return_tensors="pt", truncation=False)
        offsets = tokens.pop("offset_mapping")[0].numpy()
        if tokens["input_ids"].shape[1] > self.model.config.max_position_embeddings:
            raise ValueError("文本超过编码器最大长度；不默默截断，请实现有记录的分块策略")
        ids = tokens["input_ids"][0].numpy()
        attention = tokens["attention_mask"][0].numpy()
        with torch.inference_mode():
            hidden = self.model(**tokens.to(self.model.device)).last_hidden_state[0].float().cpu().numpy()
        features, valid, mapping = pool_tokens(hidden, offsets, [w["char_span"] for w in words], attention)
        atomic_npz(directory / "text.npz", features=features, valid=valid,
                   token_ids=ids, token_offsets=offsets, token_attention=attention)
        atomic_json(directory / "text.json", {"model": self.spec, "word_to_tokens": mapping,
            "tokens": self.tokenizer.convert_ids_to_tokens(ids.tolist()), "dimension": features.shape[1],
            "pooling": "mean of contextual tokens whose character spans overlap the original word span"})
        return {"dimension": features.shape[1], "valid_words": int(valid.sum())}


class AudioEncoder:
    def __init__(self, cfg):
        import opensmile
        from funasr import AutoModel
        self.spec = cfg["models"]["emotion"]
        path = snapshot(self.spec, ["*.yaml", "*.json", "*.pt"])
        conf = yaml.safe_load((path / "config.yaml").read_text())
        spec = conf["model_conf"]["modalities"]["audio"]["feature_encoder_spec"]
        self.receptive, self.stride = conv_geometry(spec)
        self.model = AutoModel(model=str(path), device=cfg["device"], hub="hf",
                               disable_update=True, disable_pbar=True)
        self.model.model.eval()
        self.smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                     feature_level=opensmile.FeatureLevel.LowLevelDescriptors)

    def process(self, sample, directory, cfg):
        import torch
        media = read_json(directory / "media.json")
        wave, sr = sf.read(directory / "audio.wav", dtype="float32")
        with torch.inference_mode():
            result = self.model.generate(input=wave, fs=sr, granularity="frame",
                                         extract_embedding=True, disable_pbar=True)
        emotional = np.asarray(result[0]["feats"], np.float32)
        if emotional.ndim != 2 or emotional.shape[1] != 768:
            raise ValueError(f"期待emotion2vec_base帧级[N,768]输出，实际{emotional.shape}")
        intervals = conv_intervals(len(emotional), len(wave), sr, media["audio_offset"],
                                   self.receptive, self.stride)
        lld = self.smile.process_signal(wave, sr)
        acoustic = lld.to_numpy(np.float32)
        if acoustic.shape[1] != 25:
            raise ValueError(f"eGeMAPSv02 LLD应为25维，实际{acoustic.shape}")
        acoustic_times = np.column_stack([lld.index.get_level_values(k).total_seconds().to_numpy()
                                         for k in ("start", "end")]) + media["audio_offset"]
        # Digital silence is a missing observation, not a legitimate zero-valued one.
        audible = not media["silent_audio"]
        atomic_npz(directory / "audio_features.npz", emotion=emotional,
            emotion_intervals=intervals, emotion_valid=np.isfinite(emotional).all(1) & audible,
            acoustic=acoustic, acoustic_intervals=acoustic_times,
            acoustic_valid=np.isfinite(acoustic).all(1) & audible)
        atomic_json(directory / "audio_features.json", {"model": self.spec,
            "acoustic_names": list(lld.columns), "receptive_samples": self.receptive,
            "stride_samples": self.stride, "emotion_dimension": emotional.shape[1],
            "time_note": "卷积输入感受野提供位置锚点；Transformer表示仍包含整段上下文。未把帧数均分到整段时长。",
            "rms": media["audio_rms"], "peak": media["audio_peak"], "silent_audio": media["silent_audio"]})
        return {"emotion_frames": len(emotional), "acoustic_frames": len(acoustic)}


def box_iou(box, boxes):
    boxes = np.asarray(boxes)
    overlap = np.maximum(0, np.minimum(boxes[:, 2:], box[2:]) - np.maximum(boxes[:, :2], box[:2])).prod(1)
    area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1)
    a = np.maximum(0, box[2:] - box[:2]).prod()
    return overlap / np.maximum(area + a - overlap, 1e-12)


class VisionEncoder:
    def __init__(self, cfg):
        import torch
        from openface.face_detection import FaceDetector
        from openface.multitask_model import MultitaskPredictor
        from openface.Pytorch_Retinaface.models.retinaface import RetinaFace
        self.spec = cfg["models"]["face"]
        weights = snapshot(self.spec, ["Alignment_RetinaFace.pth", "MTL_backbone.pth"])

        class CompleteCheckpointDetector(FaceDetector):
            def _load_retinaface_model(inner, path):
                # The official wrapper otherwise opens ./weights/... relative to cwd.
                # A complete RetinaFace checkpoint already includes its backbone.
                inner.cfg = {**inner.cfg, "pretrain": False}
                model = RetinaFace(cfg=inner.cfg, phase="test")
                state = torch.load(path, map_location="cpu", weights_only=True)
                state = state.get("state_dict", state)
                state = {k.removeprefix("module."): v for k, v in state.items()}
                model.load_state_dict(state, strict=True)
                return model.to(inner.device).eval()

        self.detector = CompleteCheckpointDetector(str(weights / "Alignment_RetinaFace.pth"), device=cfg["device"])
        self.predictor = MultitaskPredictor(str(weights / "MTL_backbone.pth"), device=cfg["device"])

    def process(self, sample, directory, cfg):
        import cv2
        media = read_json(directory / "media.json")
        with np.load(directory / "frames.npz", allow_pickle=False) as frames:
            pts = frames["pts"].copy()
        selected, last = [], -float("inf")
        for i, timestamp in enumerate(pts):
            if timestamp - last >= 1 / cfg["vision"]["sample_fps"] - 1e-8:
                selected.append(i)
                last = timestamp
        selected_set = set(selected)
        features, qualities, records = [], [], []
        previous_box = None
        cap = cv2.VideoCapture(sample["video"])
        if not cap.isOpened():
            raise IOError(f"OpenCV无法打开视频：{sample['video']}")
        count = 0
        try:
            with tempfile.TemporaryDirectory(prefix="q1-frame-") as temp:
                image_path = str(Path(temp) / "frame.png")
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    index = count
                    count += 1
                    if index not in selected_set:
                        continue
                    height, width = image.shape[:2]
                    scale = min(1., cfg["vision"]["max_side"] / max(height, width))
                    small = cv2.resize(image, (round(width * scale), round(height * scale))) if scale < 1 else image
                    if not cv2.imwrite(image_path, small):
                        raise IOError("无法写入临时视频帧")
                    dets, _ = self.detector.detect_faces(image_path)
                    dets = np.asarray(dets)
                    dets = dets[dets[:, 4] >= cfg["vision"]["face_threshold"]]
                    vector, quality = np.zeros(28, np.float32), 0.0
                    record = {"frame_index": index, "pts": float(pts[index]), "faces": len(dets),
                              "bbox": None, "track_reset": False, "flags": []}
                    if len(dets):
                        areas = np.maximum(0, dets[:, 2:4] - dets[:, :2]).prod(1)
                        pick = int(np.argmax(areas))
                        if previous_box is not None:
                            ious = box_iou(previous_box, dets[:, :4])
                            if ious.max() >= cfg["vision"]["track_iou_threshold"]:
                                pick = int(np.argmax(ious))
                            else:
                                record["track_reset"] = True
                        det = dets[pick]
                        bbox = det[:4].copy()
                        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, small.shape[1])
                        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, small.shape[0])
                        left, top, right, bottom = bbox.astype(int)
                        if right > left and bottom > top:
                            emotion, gaze, au = self.predictor.predict(small[top:bottom, left:right])
                            points = (det[5:15].reshape(5, 2) - bbox[:2]) / np.maximum(bbox[2:] - bbox[:2], 1)
                            vector = np.concatenate([emotion.detach().float().cpu().numpy().ravel(),
                                gaze.detach().float().cpu().numpy().ravel(),
                                au.detach().float().cpu().numpy().ravel(), points.ravel()]).astype(np.float32)
                            if vector.shape != (28,):
                                raise ValueError(f"OpenFace输出维数改变：{vector.shape}，请核验权重和包版本")
                            if np.isfinite(vector).all():
                                quality = float(det[4])
                            else:
                                record["flags"].append("nonfinite_prediction")
                                vector = np.zeros(28, np.float32)
                            previous_box = bbox
                            record["bbox"] = (bbox / scale).tolist()
                        else:
                            record["flags"].append("invalid_bbox")
                    else:
                        record["flags"].append("no_face")
                        previous_box = None
                    features.append(vector)
                    qualities.append(quality)
                    records.append(record)
        finally:
            cap.release()
        if count != len(pts) or len(records) != len(selected):
            raise ValueError(f"OpenCV解码帧数{count}与ffprobe帧数{len(pts)}不一致，拒绝错配PTS")
        times = sampled_support(pts[selected], media["video_start"], media["video_end"])
        atomic_npz(directory / "vision.npz", features=np.asarray(features, np.float32), intervals=times,
                   quality=np.asarray(qualities, np.float32), frame_indices=np.array(selected), pts=pts[selected])
        names = ([f"emotion_logit_{i}" for i in range(8)] + ["gaze_0", "gaze_1"]
                 + [f"au_raw_{i}" for i in range(8)] + [f"landmark_{i}_{c}" for i in range(5) for c in "xy"])
        atomic_json(directory / "vision.json", {"model": self.spec, "feature_names": names, "frames": records,
            "selection": "largest face initially; maximum IoU subsequently; reset if IoU below threshold",
            "limitations": "主体跟踪不等于说话人识别；5点来自RetinaFace，未启用STAR稠密关键点。AU索引不冒充FACS编号。",
            "time_rule": "nearest sampled frame piecewise-constant support; original source PTS also retained"})
        return {"sampled_frames": len(records), "valid_frames": int(np.count_nonzero(qualities))}
