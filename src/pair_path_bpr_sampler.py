from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData


Pair = Tuple[int, int]
PathTriplet = Tuple[int, int, int]
PathQuad = Tuple[int, int, int, int]
PathTuple = Tuple[int, ...]
NegativeStrategy = Literal['random', 'cross_drug', 'cross_disease', 'mixed']
SampleDict = Dict[str, Tensor]
BatchDict = Dict[str, Tensor]
EdgeType = Tuple[str, str, str]


class PairPathBPRDataset(Dataset[SampleDict]):
    """
    ?? Attention Pooling ? pair-level BPR ????

    ?????? `(drug, gene, disease)` ????????
    ?? `use_pathway_quads=True` ??????????
    `(drug, gene, pathway, disease)` ????
    """

    def __init__(
        self,
        data: HeteroData,
        ho_attr_name: str = 'ho_pos_paths',
        positive_paths: Optional[Tensor] = None,
        pair_to_paths: Optional[Mapping[Pair, Sequence[Sequence[int]]]] = None,
        known_positive_pairs: Optional[Tensor | Sequence[Sequence[int]] | Set[Pair]] = None,
        negative_strategy: NegativeStrategy = 'mixed',
        max_sampling_attempts: int = 64,
        use_pathway_quads: bool = False,
        pathway_node_type: str = 'pathway',
        pathway_dummy_global_id: int = 0,
        pathway_edge_types: Optional[Sequence[EdgeType]] = None,
    ) -> None:
        super().__init__()

        if positive_paths is None and not hasattr(data, ho_attr_name):
            raise AttributeError(f'HeteroData ?????? `{ho_attr_name}`?')

        raw_positive_paths = positive_paths if positive_paths is not None else getattr(data, ho_attr_name)
        if not isinstance(raw_positive_paths, Tensor):
            raise TypeError('`positive_paths` ? `data.ho_pos_paths` ??? torch.Tensor?')
        if raw_positive_paths.dim() != 2 or raw_positive_paths.size(1) not in {3, 4}:
            raise ValueError(
                '???????????? `[num_paths, 3]` ? `[num_paths, 4]`?'
                f'???? {tuple(raw_positive_paths.shape)}?'
            )
        if raw_positive_paths.size(0) == 0:
            raise ValueError('????????????? pair-level ???')

        self.data = data
        self.ho_attr_name = ho_attr_name
        self.negative_strategy = negative_strategy
        self.max_sampling_attempts = max_sampling_attempts
        self.use_pathway_quads = bool(use_pathway_quads)
        self.pathway_node_type = pathway_node_type
        self.pathway_dummy_global_id = int(pathway_dummy_global_id)
        self.pathway_edge_types = tuple(pathway_edge_types) if pathway_edge_types is not None else None
        self.input_path_len = int(raw_positive_paths.size(1))

        base_path_node_types = tuple(
            getattr(data, 'ho_path_node_types', ('drug', 'gene/protein', 'disease'))
        )
        self.path_node_types = self._resolve_path_node_types(base_path_node_types)
        self.path_len = len(self.path_node_types)

        self.drug_node_type = self.path_node_types[0]
        self.gene_node_type = self.path_node_types[1]
        self.disease_node_type = self.path_node_types[-1]
        self.pathway_node_type = self.path_node_types[2] if self.use_pathway_quads else pathway_node_type

        required_node_types = [self.drug_node_type, self.gene_node_type, self.disease_node_type]
        if self.use_pathway_quads:
            required_node_types.append(self.pathway_node_type)
        for node_type in required_node_types:
            if node_type not in data.node_types:
                raise KeyError(f'HeteroData ???????????? `{node_type}`?')
            if 'global_id' not in data[node_type]:
                raise KeyError(f'`data[{node_type!r}]` ??? `global_id`?')

        if self.use_pathway_quads:
            pathway_global_ids = set(int(x) for x in data[self.pathway_node_type].global_id.detach().cpu().tolist())
            if self.pathway_dummy_global_id in pathway_global_ids:
                raise ValueError(
                    'Dummy Pathway ID ??? pathway ?? ID ???'
                    f'{self.pathway_dummy_global_id}?'
                )

        self.ho_pos_paths = raw_positive_paths.detach().cpu().to(torch.long).contiguous()
        self.drug_ids: List[int] = [
            int(x) for x in data[self.drug_node_type].global_id.detach().cpu().tolist()
        ]
        self.disease_ids: List[int] = [
            int(x) for x in data[self.disease_node_type].global_id.detach().cpu().tolist()
        ]
        self.gene_to_pathways: Dict[int, Tuple[int, ...]] = self._build_gene_to_pathways()

        self.positive_pair_to_paths = self._build_positive_pair_to_paths(self.ho_pos_paths)
        self.positive_pairs: List[Pair] = list(self.positive_pair_to_paths.keys())
        self.positive_pair_set: Set[Pair] = set(self.positive_pairs)
        self.known_positive_pair_set: Set[Pair] = self._build_known_positive_pair_set(
            known_positive_pairs=known_positive_pairs,
            fallback_positive_pairs=self.positive_pair_set,
        )
        self.path_bank: Dict[Pair, Tensor] = self._build_path_bank(
            pair_to_paths=pair_to_paths,
            fallback_positive_pair_to_paths=self.positive_pair_to_paths,
        )
        self.topology_path_bank: Dict[Pair, Tensor] = self._build_topology_path_bank(
            pair_to_paths=pair_to_paths,
        )
        self.connected_pairs: List[Pair] = list(self.topology_path_bank.keys())
        self.connected_pair_set: Set[Pair] = set(self.connected_pairs)
        (
            self.disease_to_connected_drugs,
            self.drug_to_connected_diseases,
        ) = self._build_connected_pair_indices(self.topology_path_bank)

    def __len__(self) -> int:
        return len(self.positive_pairs)

    def __getitem__(self, index: int) -> SampleDict:
        pos_pair = self.positive_pairs[index]
        neg_pair = self._sample_negative_pair(pos_pair=pos_pair)

        pos_paths = self.path_bank[pos_pair]
        neg_paths = self.topology_path_bank.get(
            neg_pair,
            torch.empty((0, self.path_len), dtype=torch.long),
        )

        return {
            'pos_pair_ids': torch.tensor(pos_pair, dtype=torch.long),
            'pos_paths': pos_paths.clone(),
            'neg_pair_ids': torch.tensor(neg_pair, dtype=torch.long),
            'neg_paths': neg_paths.clone(),
        }

    def collate_fn(self, samples: Sequence[SampleDict]) -> BatchDict:
        if not samples:
            raise ValueError('`samples` ?????')

        pos_pair_ids = torch.stack([sample['pos_pair_ids'] for sample in samples], dim=0)
        neg_pair_ids = torch.stack([sample['neg_pair_ids'] for sample in samples], dim=0)

        pos_paths, pos_attention_mask = self._pad_path_tensors(
            path_tensors=[sample['pos_paths'] for sample in samples]
        )
        neg_paths, neg_attention_mask = self._pad_path_tensors(
            path_tensors=[sample['neg_paths'] for sample in samples]
        )

        return {
            'pos_pair_ids': pos_pair_ids,
            'pos_paths': pos_paths,
            'pos_attention_mask': pos_attention_mask,
            'neg_pair_ids': neg_pair_ids,
            'neg_paths': neg_paths,
            'neg_attention_mask': neg_attention_mask,
        }

    def _resolve_path_node_types(self, base_path_node_types: Sequence[str]) -> Tuple[str, ...]:
        if self.use_pathway_quads:
            if len(base_path_node_types) == 4:
                return tuple(base_path_node_types)
            if len(base_path_node_types) == 3:
                return (
                    str(base_path_node_types[0]),
                    str(base_path_node_types[1]),
                    self.pathway_node_type,
                    str(base_path_node_types[2]),
                )
            raise ValueError('?? `use_pathway_quads` ??`ho_path_node_types` ??? 3 ?? 4 ??')

        if len(base_path_node_types) != 3:
            raise ValueError(
                '?? `use_pathway_quads` ???????????'
                f'???? {tuple(base_path_node_types)}?'
            )
        return tuple(base_path_node_types)

    def _build_positive_pair_to_paths(self, ho_pos_paths: Tensor) -> Dict[Pair, Tensor]:
        grouped_paths: DefaultDict[Pair, List[PathTuple]] = defaultdict(list)
        for raw_path in ho_pos_paths.tolist():
            for expanded_path in self._expand_input_path(tuple(int(x) for x in raw_path)):
                grouped_paths[(expanded_path[0], expanded_path[-1])].append(expanded_path)

        pair_to_paths: Dict[Pair, Tensor] = {}
        for pair, path_list in grouped_paths.items():
            deduplicated_paths = list(dict.fromkeys(path_list))
            pair_to_paths[pair] = torch.tensor(deduplicated_paths, dtype=torch.long)
        return pair_to_paths

    def _build_known_positive_pair_set(
        self,
        known_positive_pairs: Optional[Tensor | Sequence[Sequence[int]] | Set[Pair]],
        fallback_positive_pairs: Set[Pair],
    ) -> Set[Pair]:
        pair_set: Set[Pair] = set(fallback_positive_pairs)
        if known_positive_pairs is None:
            return pair_set

        if isinstance(known_positive_pairs, Tensor):
            if known_positive_pairs.dim() != 2 or known_positive_pairs.size(1) not in {2, 3, 4}:
                raise ValueError(
                    '`known_positive_pairs` ?? Tensor?????? `(N, 2)`?`(N, 3)` ? `(N, 4)`?'
                )
            if known_positive_pairs.size(1) == 2:
                for pair in known_positive_pairs.detach().cpu().to(torch.long).tolist():
                    pair_set.add((int(pair[0]), int(pair[1])))
            else:
                for path in known_positive_pairs.detach().cpu().to(torch.long).tolist():
                    pair_set.add((int(path[0]), int(path[-1])))
            return pair_set

        for item in known_positive_pairs:
            if len(item) == 2:
                pair_set.add((int(item[0]), int(item[1])))
            elif len(item) in {3, 4}:
                pair_set.add((int(item[0]), int(item[-1])))
            else:
                raise ValueError('`known_positive_pairs` ????????? 2?3 ? 4?')
        return pair_set

    def _build_path_bank(
        self,
        pair_to_paths: Optional[Mapping[Pair, Sequence[Sequence[int]]]],
        fallback_positive_pair_to_paths: Mapping[Pair, Tensor],
    ) -> Dict[Pair, Tensor]:
        path_bank: Dict[Pair, Tensor] = {}

        if pair_to_paths is not None:
            for pair, path_list in pair_to_paths.items():
                normalized_pair = (int(pair[0]), int(pair[1]))
                path_bank[normalized_pair] = self._normalize_path_list(path_list)

        for pair, path_tensor in fallback_positive_pair_to_paths.items():
            if pair not in path_bank:
                path_bank[pair] = path_tensor

        return path_bank

    def _build_topology_path_bank(
        self,
        pair_to_paths: Optional[Mapping[Pair, Sequence[Sequence[int]]]],
    ) -> Dict[Pair, Tensor]:
        gene_to_drugs, gene_to_diseases = self._collect_gene_connectivity_from_graph()
        topology_buffer: DefaultDict[Pair, List[PathTuple]] = defaultdict(list)

        shared_gene_ids = set(gene_to_drugs.keys()).intersection(gene_to_diseases.keys())
        for gene_id in shared_gene_ids:
            unique_drugs = list(dict.fromkeys(gene_to_drugs[gene_id]))
            unique_diseases = list(dict.fromkeys(gene_to_diseases[gene_id]))
            pathway_ids = self._expand_gene_bridge_path(gene_id=gene_id)
            for drug_id in unique_drugs:
                for disease_id in unique_diseases:
                    if self.use_pathway_quads:
                        for pathway_id in pathway_ids:
                            topology_buffer[(drug_id, disease_id)].append(
                                (drug_id, gene_id, pathway_id, disease_id)
                            )
                    else:
                        topology_buffer[(drug_id, disease_id)].append((drug_id, gene_id, disease_id))

        topology_path_bank: Dict[Pair, Tensor] = {}
        for pair, path_list in topology_buffer.items():
            deduplicated_paths = list(dict.fromkeys(path_list))
            topology_path_bank[pair] = torch.tensor(deduplicated_paths, dtype=torch.long)

        if pair_to_paths is not None:
            for pair, path_list in pair_to_paths.items():
                normalized_pair = (int(pair[0]), int(pair[1]))
                external_paths = self._normalize_path_list(path_list)
                if normalized_pair not in topology_path_bank:
                    topology_path_bank[normalized_pair] = external_paths
                    continue

                merged_paths = torch.cat([topology_path_bank[normalized_pair], external_paths], dim=0)
                unique_paths = list(dict.fromkeys(tuple(int(x) for x in row) for row in merged_paths.tolist()))
                topology_path_bank[normalized_pair] = torch.tensor(unique_paths, dtype=torch.long)

        return topology_path_bank

    def _collect_gene_connectivity_from_graph(
        self,
    ) -> Tuple[DefaultDict[int, List[int]], DefaultDict[int, List[int]]]:
        gene_to_drugs: DefaultDict[int, List[int]] = defaultdict(list)
        gene_to_diseases: DefaultDict[int, List[int]] = defaultdict(list)

        for edge_type, edge_index in self.data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if edge_index.numel() == 0:
                continue

            edge_index_cpu = edge_index.detach().cpu().to(torch.long)
            src_global_ids = self.data[src_type].global_id.detach().cpu()[edge_index_cpu[0]]
            dst_global_ids = self.data[dst_type].global_id.detach().cpu()[edge_index_cpu[1]]

            if src_type == self.drug_node_type and dst_type == self.gene_node_type:
                for drug_id, gene_id in zip(src_global_ids.tolist(), dst_global_ids.tolist()):
                    gene_to_drugs[int(gene_id)].append(int(drug_id))
                continue

            if src_type == self.gene_node_type and dst_type == self.drug_node_type:
                for gene_id, drug_id in zip(src_global_ids.tolist(), dst_global_ids.tolist()):
                    gene_to_drugs[int(gene_id)].append(int(drug_id))
                continue

            if src_type == self.gene_node_type and dst_type == self.disease_node_type:
                for gene_id, disease_id in zip(src_global_ids.tolist(), dst_global_ids.tolist()):
                    gene_to_diseases[int(gene_id)].append(int(disease_id))
                continue

            if src_type == self.disease_node_type and dst_type == self.gene_node_type:
                for disease_id, gene_id in zip(src_global_ids.tolist(), dst_global_ids.tolist()):
                    gene_to_diseases[int(gene_id)].append(int(disease_id))
                continue

        return gene_to_drugs, gene_to_diseases

    def _build_gene_to_pathways(self) -> Dict[int, Tuple[int, ...]]:
        if not self.use_pathway_quads:
            return {}

        gene_to_pathways_buffer: DefaultDict[int, List[int]] = defaultdict(list)
        candidate_edge_types: Sequence[EdgeType]
        if self.pathway_edge_types is not None:
            candidate_edge_types = self.pathway_edge_types
        else:
            candidate_edge_types = tuple(self.data.edge_index_dict.keys())

        for edge_type in candidate_edge_types:
            if edge_type not in self.data.edge_index_dict:
                continue

            src_type, _, dst_type = edge_type
            if {src_type, dst_type} != {self.gene_node_type, self.pathway_node_type}:
                continue

            edge_index = self.data.edge_index_dict[edge_type]
            if edge_index.numel() == 0:
                continue

            edge_index_cpu = edge_index.detach().cpu().to(torch.long)
            src_global_ids = self.data[src_type].global_id.detach().cpu()[edge_index_cpu[0]]
            dst_global_ids = self.data[dst_type].global_id.detach().cpu()[edge_index_cpu[1]]

            if src_type == self.gene_node_type and dst_type == self.pathway_node_type:
                pairs = zip(src_global_ids.tolist(), dst_global_ids.tolist())
            else:
                pairs = zip(dst_global_ids.tolist(), src_global_ids.tolist())

            for gene_id, pathway_id in pairs:
                gene_to_pathways_buffer[int(gene_id)].append(int(pathway_id))

        gene_to_pathways: Dict[int, Tuple[int, ...]] = {}
        for gene_id, pathway_ids in gene_to_pathways_buffer.items():
            gene_to_pathways[gene_id] = tuple(dict.fromkeys(pathway_ids))
        return gene_to_pathways

    def _build_connected_pair_indices(
        self,
        topology_path_bank: Mapping[Pair, Tensor],
    ) -> Tuple[Dict[int, Tuple[int, ...]], Dict[int, Tuple[int, ...]]]:
        disease_to_connected_drugs_set: DefaultDict[int, Set[int]] = defaultdict(set)
        drug_to_connected_diseases_set: DefaultDict[int, Set[int]] = defaultdict(set)

        for drug_id, disease_id in topology_path_bank.keys():
            disease_to_connected_drugs_set[disease_id].add(drug_id)
            drug_to_connected_diseases_set[drug_id].add(disease_id)

        disease_to_connected_drugs = {
            disease_id: tuple(sorted(drug_ids))
            for disease_id, drug_ids in disease_to_connected_drugs_set.items()
        }
        drug_to_connected_diseases = {
            drug_id: tuple(sorted(disease_ids))
            for drug_id, disease_ids in drug_to_connected_diseases_set.items()
        }
        return disease_to_connected_drugs, drug_to_connected_diseases

    def _expand_input_path(self, raw_path: Sequence[int]) -> List[PathTuple]:
        normalized_path = tuple(int(x) for x in raw_path)

        if not self.use_pathway_quads:
            if len(normalized_path) != 3:
                raise ValueError(
                    '???????????????????? 3?'
                    f'{normalized_path}?'
                )
            return [normalized_path]

        if len(normalized_path) == 4:
            return [normalized_path]
        if len(normalized_path) != 3:
            raise ValueError(
                '?? pathway ?????????????? 3 ? 4?'
                f'???? {normalized_path}?'
            )

        drug_id, gene_id, disease_id = normalized_path
        pathway_ids = self._expand_gene_bridge_path(gene_id=gene_id)
        return [(drug_id, gene_id, pathway_id, disease_id) for pathway_id in pathway_ids]

    def _expand_gene_bridge_path(self, gene_id: int) -> Tuple[int, ...]:
        if not self.use_pathway_quads:
            return ()
        pathway_ids = self.gene_to_pathways.get(int(gene_id), ())
        if pathway_ids:
            return pathway_ids
        return (self.pathway_dummy_global_id,)

    def _normalize_path_list(self, path_list: Sequence[Sequence[int]]) -> Tensor:
        if len(path_list) == 0:
            return torch.empty((0, self.path_len), dtype=torch.long)

        normalized_paths: List[PathTuple] = []
        for path in path_list:
            normalized_paths.extend(self._expand_input_path(path))

        deduplicated_paths = list(dict.fromkeys(normalized_paths))
        return torch.tensor(deduplicated_paths, dtype=torch.long)

    def _sample_negative_pair(self, pos_pair: Pair) -> Pair:
        candidate_strategies: List[NegativeStrategy]
        if self.negative_strategy == 'mixed':
            strategy_pool: List[NegativeStrategy] = ['cross_drug', 'cross_disease', 'random']
            first_choice = strategy_pool[
                torch.randint(low=0, high=len(strategy_pool), size=(1,)).item()
            ]
            candidate_strategies = [first_choice] + [s for s in strategy_pool if s != first_choice]
        elif self.negative_strategy == 'cross_drug':
            candidate_strategies = ['cross_drug', 'random']
        elif self.negative_strategy == 'cross_disease':
            candidate_strategies = ['cross_disease', 'random']
        else:
            candidate_strategies = [self.negative_strategy]

        for strategy in candidate_strategies:
            neg_pair = self._try_sample_negative_pair(pos_pair=pos_pair, strategy=strategy)
            if neg_pair is not None:
                return neg_pair

        raise RuntimeError(
            f'??????????? pair={pos_pair} ???????strategy={self.negative_strategy}?'
        )

    def _try_sample_negative_pair(
        self,
        pos_pair: Pair,
        strategy: NegativeStrategy,
    ) -> Optional[Pair]:
        pos_drug_id, pos_disease_id = pos_pair

        if strategy == 'random':
            connected_candidate = self._try_sample_topology_random_negative(pos_pair=pos_pair)
            if connected_candidate is not None:
                return connected_candidate

            for _ in range(self.max_sampling_attempts):
                candidate = (
                    self.drug_ids[torch.randint(low=0, high=len(self.drug_ids), size=(1,)).item()],
                    self.disease_ids[
                        torch.randint(low=0, high=len(self.disease_ids), size=(1,)).item()
                    ],
                )
                if candidate != pos_pair and candidate not in self.known_positive_pair_set:
                    return candidate

            for drug_id in self.drug_ids:
                for disease_id in self.disease_ids:
                    candidate = (drug_id, disease_id)
                    if candidate != pos_pair and candidate not in self.known_positive_pair_set:
                        return candidate
            return None

        if strategy == 'cross_drug':
            candidate_drug_ids = self.disease_to_connected_drugs.get(pos_disease_id, ())
            return self._sample_connected_cross_pair(
                variable_candidates=candidate_drug_ids,
                fixed_id=pos_disease_id,
                pos_pair=pos_pair,
                mode='cross_drug',
            )

        if strategy == 'cross_disease':
            candidate_disease_ids = self.drug_to_connected_diseases.get(pos_drug_id, ())
            return self._sample_connected_cross_pair(
                variable_candidates=candidate_disease_ids,
                fixed_id=pos_drug_id,
                pos_pair=pos_pair,
                mode='cross_disease',
            )

        raise ValueError(f'???? negative strategy: {strategy}')

    def _try_sample_topology_random_negative(self, pos_pair: Pair) -> Optional[Pair]:
        if not self.connected_pairs:
            return None

        for _ in range(self.max_sampling_attempts):
            candidate = self.connected_pairs[
                torch.randint(low=0, high=len(self.connected_pairs), size=(1,)).item()
            ]
            if candidate != pos_pair and candidate not in self.known_positive_pair_set:
                return candidate

        for candidate in self.connected_pairs:
            if candidate != pos_pair and candidate not in self.known_positive_pair_set:
                return candidate
        return None

    def _sample_connected_cross_pair(
        self,
        variable_candidates: Sequence[int],
        fixed_id: int,
        pos_pair: Pair,
        mode: Literal['cross_drug', 'cross_disease'],
    ) -> Optional[Pair]:
        if not variable_candidates:
            return None

        for _ in range(self.max_sampling_attempts):
            variable_id = variable_candidates[
                torch.randint(low=0, high=len(variable_candidates), size=(1,)).item()
            ]
            candidate = (
                (int(variable_id), int(fixed_id))
                if mode == 'cross_drug'
                else (int(fixed_id), int(variable_id))
            )
            if candidate != pos_pair and candidate not in self.known_positive_pair_set:
                return candidate

        for variable_id in variable_candidates:
            candidate = (
                (int(variable_id), int(fixed_id))
                if mode == 'cross_drug'
                else (int(fixed_id), int(variable_id))
            )
            if candidate != pos_pair and candidate not in self.known_positive_pair_set:
                return candidate
        return None

    def _pad_path_tensors(self, path_tensors: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
        batch_size = len(path_tensors)
        max_num_paths = max(max(int(path_tensor.size(0)), 1) for path_tensor in path_tensors)

        padded_paths = torch.zeros((batch_size, max_num_paths, self.path_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_num_paths), dtype=torch.bool)

        for sample_index, path_tensor in enumerate(path_tensors):
            if path_tensor.dim() != 2 or path_tensor.size(1) != self.path_len:
                raise ValueError(
                    '????? `paths` ??????? `(num_paths, path_len)`?'
                    f'???? {tuple(path_tensor.shape)}?'
                )

            num_real_paths = int(path_tensor.size(0))
            if num_real_paths == 0:
                continue

            padded_paths[sample_index, :num_real_paths] = path_tensor
            attention_mask[sample_index, :num_real_paths] = True

        return padded_paths, attention_mask


def build_pair_path_bpr_dataloader(
    data: HeteroData,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    ho_attr_name: str = 'ho_pos_paths',
    positive_paths: Optional[Tensor] = None,
    pair_to_paths: Optional[Mapping[Pair, Sequence[Sequence[int]]]] = None,
    known_positive_pairs: Optional[Tensor | Sequence[Sequence[int]] | Set[Pair]] = None,
    negative_strategy: NegativeStrategy = 'mixed',
    max_sampling_attempts: int = 64,
    use_pathway_quads: bool = False,
    pathway_node_type: str = 'pathway',
    pathway_dummy_global_id: int = 0,
    pathway_edge_types: Optional[Sequence[EdgeType]] = None,
) -> DataLoader[BatchDict]:
    dataset = PairPathBPRDataset(
        data=data,
        ho_attr_name=ho_attr_name,
        positive_paths=positive_paths,
        pair_to_paths=pair_to_paths,
        known_positive_pairs=known_positive_pairs,
        negative_strategy=negative_strategy,
        max_sampling_attempts=max_sampling_attempts,
        use_pathway_quads=use_pathway_quads,
        pathway_node_type=pathway_node_type,
        pathway_dummy_global_id=pathway_dummy_global_id,
        pathway_edge_types=pathway_edge_types,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=dataset.collate_fn,
    )
