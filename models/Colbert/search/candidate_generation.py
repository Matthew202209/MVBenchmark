import torch

from models.Colbert.search.strided_tensor import StridedTensor
from .strided_tensor_core import _create_mask, _create_view


class CandidateGeneration:

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

    def get_cells(self, Q, ncells):
        if self.use_gpu:
            scores = (self.codec.centroids @ Q.T)
        else:
            scores = (self.codec.centroids.to("cpu").to(torch.float32) @ Q.T)
        if ncells == 1:
            cells = scores.argmax(dim=0, keepdim=True).permute(1, 0)
        else:
            cells = scores.content_topk(ncells, dim=0, sorted=False).indices.permute(1, 0)  # (32, ncells)
        #import pdb; pdb.set_trace()
        cells = cells.flatten().contiguous()  # (32 * ncells,)
        cells = cells.unique(sorted=False)
        return cells, scores

    def generate_candidate_eids(self, Q, ncells):
        cells, scores = self.get_cells(Q, ncells)

        eids, cell_lengths = self.ivf.lookup(cells)  # eids = (packedlen,)  lengths = (32 * ncells,)
        eids = eids.long()
        if self.use_gpu:
            eids = eids.cuda()
        return eids, scores

    def generate_candidate_eids_colbertv2(self, Q, nprobe):

        if self.use_gpu:
            cells = (self.codec.centroids @ Q.T).content_topk(nprobe, dim=0, sorted=False).indices.permute(1, 0)  # (32, nprobe)
        else:
            cells = (self.codec.centroids.to("cpu").to(torch.float32) @ Q.T).content_topk(nprobe, dim=0, sorted=False).indices.permute(1, 0)  # (32, nprobe)

        cells = cells.flatten().contiguous()  # (32 * nprobe,)
        cells = cells.unique(sorted=False)

        eids, cell_lengths = self.ivf_ori.lookup_colberv2(cells)  # eids = (packedlen,)  lengths = (32 * nprobe,)

        eids = eids.long()
        if self.use_gpu:
            eids = eids.cuda()
        return eids

    def generate_candidate_pids(self, Q, ncells):
        cells, scores = self.get_cells(Q, ncells)

        pids, cell_lengths = self.ivf.lookup(cells)
        if self.use_gpu:
            pids = pids.cuda()
        return pids, scores

    def generate_candidate_scores(self, Q, eids):
        E = self.lookup_eids(eids)
        if self.use_gpu:
            E = E.cuda()
        return (Q.unsqueeze(0) @ E.unsqueeze(2)).squeeze(-1).T

    def generate_candidates(self, config, Q):
        ncells = config.ncells

        assert isinstance(self.ivf, StridedTensor)

        Q = Q.squeeze(0)
        if self.use_gpu:
            Q = Q.cuda().half()
        assert Q.dim() == 2

        pids, centroid_scores = self.generate_candidate_pids(Q, ncells)

        sorter = pids.sort()
        pids = sorter.values

        pids, pids_counts = torch.unique_consecutive(pids, return_counts=True)
        if self.use_gpu:
            pids, pids_counts = pids.cuda(), pids_counts.cuda()

        return pids, centroid_scores

    def generate_candidates_vanilla_colbertv2(self, config, Q):
        nprobe = config.nprobe
        ncandidates = config.ncandidates

        # assert isinstance(self.ivf, StridedTensor)
        Q = Q.squeeze(0)
        if self.use_gpu:
            Q = Q.cuda().half()
        assert Q.dim() == 2

        eids = self.generate_candidate_eids_colbertv2(Q, nprobe)
        eids = torch.unique(eids, sorted=False)
        pids = self.emb2pid[eids.long()]

        if self.use_gpu:
            pids = pids.cuda()
        sorter = pids.sort()
        pids = sorter.values

        pids, pids_counts = torch.unique_consecutive(pids, return_counts=True)

        if self.use_gpu:
            pids, pids_counts = pids.cuda(), pids_counts.cuda()

        if len(pids) <= ncandidates:
            return pids

        pids_offsets = pids_counts.cumsum(dim=0) - pids_counts[0]
        eids = eids[sorter.indices]
        scores = self.generate_candidate_scores(Q, eids)

        stride = pids_counts.max().item()

        scores_dim1, scores_dim2 = scores.size()
        scores = scores.flatten()
        scores = torch.nn.functional.pad(scores, (0, stride))
        if self.use_gpu:
            scores = scores.cuda()

        pids_offsets2 = pids_offsets.repeat(scores_dim1, 1)
        if self.use_gpu:
            pids_offsets2 += torch.arange(scores_dim1).cuda().unsqueeze(1) * scores_dim2
        else:
            pids_offsets2 += torch.arange(scores_dim1).unsqueeze(1) * scores_dim2
        pids_offsets2 = pids_offsets2.flatten()

        pids_counts2 = pids_counts.repeat(scores_dim1, 1).flatten()

        scores_padded = _create_view(scores, stride, [])[pids_offsets2] * _create_mask(pids_counts2, stride, use_gpu=self.use_gpu)
        scores_maxsim = scores_padded.max(-1).values

        num_pids = pids.size(0)
        pids = pids.unsqueeze(0).repeat(scores_dim1, 1).flatten()
        scores = scores_maxsim

        assert pids.size() == scores.size()

        if self.use_gpu:
            indices = torch.arange(0, scores_dim1).cuda() * num_pids
        else:
            indices = torch.arange(0, scores_dim1) * num_pids
        indices = indices.repeat(num_pids, 1)
        if self.use_gpu:
            indices += torch.arange(num_pids).cuda().unsqueeze(1)
        else:
            indices += torch.arange(num_pids).unsqueeze(1)
        indices = indices.flatten()

        pids = pids[indices]
        scores = scores[indices]

        pids, pids_counts = torch.unique_consecutive(pids, return_counts=True)
        pids_offsets = pids_counts.cumsum(dim=0) - pids_counts[0]
        stride = pids_counts.max().item()

        scores = torch.nn.functional.pad(scores, (0, stride))
        scores_padded = _create_view(scores, stride, [])[pids_offsets] * _create_mask(pids_counts, stride, use_gpu=self.use_gpu)
        scores_lb = scores_padded.sum(-1)

        assert scores_lb.size(0) == pids.size(0)

        if scores_lb.size(0) > ncandidates:
            pids = pids[scores_lb.content_topk(ncandidates, dim=-1, sorted=True).indices]

        return pids
