"""Frozen multilingual text and visible target geometry for manipulation policies."""

from hashlib import sha256
from math import cos, sin
from pathlib import Path

import numpy as np


class FrozenMiniLM:
    """Pinned, offline CPU encoder. Cache the output before entering a physics rollout."""

    model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    revision = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    dimension = 384
    pooling = "mean-l2"
    _files = {
        "tokenizer.json": "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8",
        "onnx/model_quint8_avx2.onnx": "98a01d88b7de996cdea58c32ca71208c09968d143798814b2ea09d3439dc334f",
    }

    def __init__(self, directory):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        contents = {}
        for name, digest in self._files.items():
            data = (Path(directory) / name).read_bytes()
            if sha256(data).hexdigest() != digest:
                raise ValueError(f"MiniLM asset digest differs: {name}")
            contents[name] = data
        self._tokenizer = Tokenizer.from_buffer(contents["tokenizer.json"])
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self._model = ort.InferenceSession(
            contents["onnx/model_quint8_avx2.onnx"], options, providers=["CPUExecutionProvider"]
        )
        self._inputs = tuple(i.name for i in self._model.get_inputs())

    def encode(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("expected nonempty instruction or target label")
        tokens = self._tokenizer.encode(text)
        if len(tokens.ids) > 128:
            raise ValueError("instruction exceeds the model's 128-token sentence limit")
        values = dict(
            input_ids=tokens.ids,
            attention_mask=tokens.attention_mask,
            token_type_ids=tokens.type_ids,
        )
        inputs = {k: np.asarray([values[k]], dtype=np.int64) for k in self._inputs}
        (hidden,) = self._model.run(["last_hidden_state"], inputs)
        mask = np.asarray(tokens.attention_mask, dtype=bool)
        pooled = hidden[0, mask].mean(0)
        norm = np.linalg.norm(pooled)
        if pooled.shape != (self.dimension,) or not np.isfinite(norm) or norm <= 0:
            raise ValueError("invalid sentence representation")
        return pooled / norm


class GoalLanguageTable:
    """Precompute model inputs from instructions and every visible candidate in authored order."""

    def __init__(self, encoder, scenes, world_keys, device, language):
        import torch

        if language not in ("zh", "en"):
            raise ValueError("instruction language must be zh or en")
        self._language = language
        self._keys = tuple(world_keys)
        texts = {}
        catalogs = []
        for k in self._keys:
            scene = scenes[k]
            for goal in scene.goals:
                for text in self._variants(goal):
                    texts[text] = None
            labels = []
            for target in scene.targets:
                label = getattr(target, f"label_{language}")
                texts[label] = None
                labels.append(
                    (
                        label,
                        (*target.position, sin(target.yaw), cos(target.yaw), *target.half_extents),
                    )
                )
            catalogs.append(labels)

        vectors = np.stack([encoder.encode(text) for text in texts])
        if (
            vectors.shape != (len(texts), encoder.dimension)
            or vectors.dtype != np.float32
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("encoder must return finite float32 sentence vectors")
        address = {text: i for i, text in enumerate(texts)}
        self._variants_gpu = {}
        for k in self._keys:
            for goal in scenes[k].goals:
                variants = self._variants(goal)
                if variants not in self._variants_gpu:
                    self._variants_gpu[variants] = torch.tensor(
                        vectors[[address[text] for text in variants]], device=device
                    )
        width = max(map(len, catalogs))
        targets = np.zeros((len(catalogs), width, encoder.dimension + 7), dtype=np.float32)
        valid = np.zeros((len(catalogs), width), dtype=bool)
        for row, candidates in enumerate(catalogs):
            for col, (label, pose) in enumerate(candidates):
                targets[row, col, : encoder.dimension] = vectors[address[label]]
                targets[row, col, encoder.dimension :] = pose
                valid[row, col] = True
        self._targets = torch.tensor(targets, device=device)
        self._mask = torch.tensor(valid, device=device)
        self.binding = dict(
            schema="manipulation-visible-goals-v1",
            model=encoder.model_id,
            revision=encoder.revision,
            language=language,
            dimension=encoder.dimension,
            catalog_vectors_sha256=sha256(vectors.tobytes()).hexdigest(),
            pose=("x", "y", "z", "sin_yaw", "cos_yaw", "half_x", "half_y"),
            paraphrases="episode-modulo-count",
            pooling=encoder.pooling,
        )

    def _variants(self, goal):
        return getattr(goal, f"instructions_{self._language}")

    def observe(self, goals, episodes):
        import torch

        rows = []
        for k in self._keys:
            bank = self._variants_gpu[self._variants(goals[k])]
            rows.append(bank[episodes[k] % len(bank)])
        return dict(
            instruction=torch.stack(rows),
            visible_targets=self._targets.clone(),
            visible_target_mask=self._mask.clone(),
        )
