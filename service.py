import os
from typing import Literal, cast

import bentoml
import numpy as np
from FlagEmbedding import BGEM3FlagModel

MODEL_PATH = os.environ.get("MODEL_PATH", "BAAI/bge-m3")

MultiVectorReturn = list[
    dict[Literal["dense_vecs", "lexical_weights", "colbert_vecs"], list[float] | dict[str, float]]
]


@bentoml.service
class BgeEmbedService:
    def __init__(self) -> None:
        self.model = BGEM3FlagModel(model_name_or_path=MODEL_PATH, batch_size=32)

    @bentoml.api
    async def dense_embed(self, sentences: list[str]) -> np.ndarray:
        resp = self.model.encode(
            sentences,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"]
        return resp

    @bentoml.api
    async def sparse_embed(self, sentences: list[str]) -> list[dict[str, float]]:
        resp = self.model.encode(
            sentences,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"]
        return resp

    @bentoml.api
    async def colbert_embed(self, sentences: list[str]) -> np.ndarray:
        resp = self.model.encode(
            sentences,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )["colbert_vecs"]
        return resp

    @bentoml.api
    async def embed(
        self,
        sentences: list[str],
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert_vecs: bool = True,
    ) -> MultiVectorReturn:
        embeddings = self.model.encode(
            sentences,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert_vecs,
        )
        # reorder the response to match the input order
        ret = []
        for i in range(len(sentences)):
            x = {}
            if return_dense:
                x["dense_vecs"] = embeddings["dense_vecs"][i]
            if return_sparse:
                x["lexical_weights"] = embeddings["lexical_weights"][i]
            if return_colbert_vecs:
                x["colbert_vecs"] = embeddings["colbert_vecs"][i][0]
            ret.append(x)
        return cast(MultiVectorReturn, ret)
