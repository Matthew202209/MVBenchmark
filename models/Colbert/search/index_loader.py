import os
import ujson
import torch
import numpy as np
import tqdm

from models.Colbert.utils.utils import lengths2offsets, print_message, dotdict, flatten
from models.Colbert.indexing.codecs.residual import ResidualCodec
from models.Colbert.indexing.utils import optimize_ivf
from models.Colbert.search.strided_tensor import StridedTensor


class IndexLoader:
    def __init__(self, index_path, use_gpu=True, is_colbertv2 = False):
        self.index_path = index_path
        self.use_gpu = use_gpu
        self.is_colbertv2 = is_colbertv2
        self.codec = None
        self.ivf = None
        self.ivf_ori = None
        self.doclens = None
        self.embeddings = None


        self._load_codec()
        self._load_ivf()
        self._load_doclens()
        self._load_embeddings()



    def _load_codec(self):
        print_message(f"#> Loading codec...")
        self.codec = ResidualCodec.load(self.index_path)

    def _load_ivf(self):
        print_message(f"#> Loading IVF...")
        if self.is_colbertv2:
            ivf_ori, ivf_ori_lengths = torch.load(os.path.join(self.index_path, "ivf.ori.pt"), map_location='cpu')
            ivf_ori = StridedTensor(ivf_ori, ivf_ori_lengths, use_gpu=self.use_gpu)
            self.ivf_ori = ivf_ori
        else:
            ivf, ivf_lengths = torch.load(os.path.join(self.index_path, "ivf.pid.pt"), map_location='cpu')
            ivf = StridedTensor(ivf, ivf_lengths, use_gpu=self.use_gpu)
            self.ivf = ivf


    def _load_doclens(self):
        doclens = []

        print_message("#> Loading doclens...")

        for chunk_idx in tqdm.tqdm(range(self.num_chunks)):
            with open(os.path.join(self.index_path, f'doclens.{chunk_idx}.json')) as f:
                chunk_doclens = ujson.load(f)
                doclens.extend(chunk_doclens)

        self.doclens = torch.tensor(doclens)

    def _load_embeddings(self):
        self.embeddings = ResidualCodec.Embeddings.load_chunks(self.index_path, range(self.num_chunks),
                                                               self.num_embeddings)

    @property
    def metadata(self):
        try:
            self._metadata
        except:
            with open(os.path.join(self.index_path, 'metadata.json')) as f:
                self._metadata = ujson.load(f)

        return self._metadata

    @property
    def config(self):
        raise NotImplementedError()  # load from dict at metadata['config']

    @property
    def num_chunks(self):
        # EVENTUALLY: If num_chunks doesn't exist (i.e., old index), fall back to counting doclens.*.json files.
        return self.metadata['num_chunks']

    @property
    def num_embeddings(self):
        # EVENTUALLY: If num_embeddings doesn't exist (i.e., old index), sum the values in doclens.*.json files.
        return self.metadata['num_embeddings']

