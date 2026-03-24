from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from collections import defaultdict
import networkx as nx
import csv

from uncha import lorentz as L
from uncha.evaluation.catalog import DatasetCatalog
from uncha.evaluation.class_names import CLASS_NAMES
from uncha.models import UNCHA, HyCoCLIP, MERU, CLIPBaseline
from uncha.tokenizer import Tokenizer
from uncha.evaluation.classification import ZeroShotClassificationEvaluator, _encode_dataset
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn


IMAGENET_SYNSET_ORDER = 'assets/imagenet_synset/all_synsets.pkl'
IMAGENET_WORDNET_GRAPH = 'assets/imagenet_synset/imagenet_isa.txt'
ANCESTOR_INDICES = 'assets/imagenet_synset/all_ancestors_indices.pkl'


def def_value():
    return "Not Present"

def create_nx_graph_from_edges(edge_file):
    """
    Create a NetworkX DiGraph object from a file containing edges of the graph.
    """
    synset_map = {}

    for s in wn.all_synsets():
        synset_map[s.offset()] = s.name().split('.')[0]
    

    graph = nx.DiGraph()
    edge_dict = defaultdict(def_value)
    synset_to_label = defaultdict(def_value)


    with open(edge_file, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            edge_dict.setdefault(row[0], []).append(row[1])
            if row[0] not in list(graph.nodes):
                graph.add_node(row[0])
                synset_to_label[row[0]] = synset_map[int(row[0][1:])]
            if row[1] not in list(graph.nodes):
                graph.add_node(row[1])
                synset_to_label[row[1]] = synset_map[int(row[1][1:])]

    for parent, children in edge_dict.items():
      for child in children:
        graph.add_edge(parent, child)

    return graph


def hierarchical_based_metrics(predicted_labels, true_labels, ancestor_indices, graph, synsets_ordering):
    """
    Calculate Hierarchical metrics
    """
    undirected_graph = graph.to_undirected()
    tree_induced_error = 0
    least_common_ancestor = 0
    jaccard = 0
    hierarchical_precision = 0
    hierarchical_recall = 0
    
    for _b in range(predicted_labels.size(0)):
        pred_label = predicted_labels[_b]
        pred_synset = synsets_ordering[pred_label]
        true_label = true_labels[_b]
        true_synset = synsets_ordering[true_label]
        pred_ancestors = ancestor_indices[pred_label]
        true_ancestors = ancestor_indices[true_label]
        pred_ancestors = set(pred_ancestors)
        true_ancestors = set(true_ancestors)

        intersection = pred_ancestors.intersection(true_ancestors)
        union = pred_ancestors.union(true_ancestors)

        tree_induced_error += nx.shortest_path_length(undirected_graph, source=pred_synset, target=true_synset)
        least_common_ancestor += len(pred_ancestors) - len(intersection) + 1 # +1 for the actual class label
        jaccard += len(intersection) / len(union)
        hierarchical_precision += len(intersection) / len(pred_ancestors)
        hierarchical_recall += len(intersection) / len(true_ancestors)
        
    return tree_induced_error, least_common_ancestor, jaccard, hierarchical_precision, hierarchical_recall, predicted_labels.size(0)


class HierarchicalMetricsEvaluator(ZeroShotClassificationEvaluator):
    """
    Evaluate model performance on Hierarchical metrics:
    1. Tree Induced Error (TIE)
    2. Least Common Ancestor (LCA)
    3. Jaccard Similarity
    4. Hierarchical Precision
    5. Hierarchical Recall
    """

    def __init__(
        self,
        datasets_and_prompts: dict[str, list[str]],
        data_dir: str | Path,
        image_size: int = 224,
    ):
        """
        Args:
            datasets_and_prompts: Dictionary mapping between dataset name and
                a list of prompt templates to fill using its class names. Add
                a single `{}` in prompt to fill with class name. Datasets
                should be among supported datasets in `DatasetCatalog`.
            data_dir: Path to directory containing sub-directories of all datasets
                that are supported by the dataset catalog.
            image_size: Resize and crop images to this size for evaluation. We
                resize the smaller image edge (keeping aspect ratio same) using
                bicubic interpolation, and take a square center crop.
        """
        super().__init__(datasets_and_prompts, data_dir, image_size)

    @torch.inference_mode()
    def __call__(self, model: UNCHA | HyCoCLIP | MERU | CLIPBaseline) -> dict[str, float]:
        model = model.eval()
        tokenizer = Tokenizer()

        imagenet_synset_order = pickle.load(open(IMAGENET_SYNSET_ORDER, 'rb'))
        ancestor_indices = pickle.load(open(ANCESTOR_INDICES, 'rb'))
        graph = create_nx_graph_from_edges(IMAGENET_WORDNET_GRAPH)


       

        results_dict = {}

        for dname, prompts in self._datasets_and_prompts.items():
            class_names = CLASS_NAMES[dname]

            # Collect text features of each class.
            all_class_feats: list[torch.Tensor] = []

            for i, name in enumerate(class_names):
                class_prompts = [_pt.format(name) for _pt in prompts]

                class_prompt_tokens = tokenizer(class_prompts)
                class_feats = model.encode_text(class_prompt_tokens, project=False)

                if isinstance(model, (UNCHA, HyCoCLIP, MERU)):
                    # Ensemble in the tangent space, then project to Hyperboloid.
                    class_feats = class_feats.mean(dim=0)
                    class_feats = class_feats * model.textual_alpha.exp()
                    class_feats = L.exp_map0(class_feats, model.curv.exp())
                else:
                    # Ensemble prompt features: normalize -> average -> normalize.
                    class_feats = F.normalize(class_feats, dim=-1)
                    class_feats = class_feats.mean(dim=0)
                    class_feats = F.normalize(class_feats, dim=-1)

                all_class_feats.append(class_feats)

            # shape: (num_classes, embed_dim)
            classifier = torch.stack(all_class_feats, dim=0)

            # Extract image features and labels from the test split of required dataset.
            loader = DataLoader(
                DatasetCatalog.build(
                    dname, self._data_dir, "test", self._image_transform
                ),
                batch_size=128,
            )
  
            image_feats, labels = _encode_dataset(loader, model, project=True)


            # Features returned by this function will be on CPU, move to device:
            image_feats = image_feats.to(model.device)

            # Measure model performance according to accuracy metric of the dataset.
            total_tie = 0
            total_lca = 0
            total_jaccard = 0
            total_hier_precision = 0
            total_hier_recall = 0
            total_val_size = 0

            # Evaluate in small batches of 256 instances.
            for _feats, _labels in tqdm(zip(image_feats.split(256), labels.split(256)), desc=f"Calculating scores"):
                # Compute pairwise similarity depending on model type:
                if isinstance(model, (UNCHA, HyCoCLIP, MERU)):
                    scores = L.pairwise_inner(_feats, classifier, model.curv.exp())
                else:
                    scores = _feats @ classifier.T

                predicted_labels = scores.argmax(dim=-1)
                tree_induced_error, least_common_ancestor, jaccard, hierarchical_precision, hierarchical_recall, b_size = hierarchical_based_metrics(
                    predicted_labels, 
                    _labels, 
                    ancestor_indices, 
                    graph, 
                    imagenet_synset_order
                    )
                total_tie += tree_induced_error
                total_lca += least_common_ancestor
                total_jaccard += jaccard
                total_hier_precision += hierarchical_precision
                total_hier_recall += hierarchical_recall

                total_val_size += b_size
            
            avg_tie = total_tie / total_val_size
            avg_lca = total_lca / total_val_size
            avg_jaccard = total_jaccard / total_val_size
            avg_hier_precision = total_hier_precision / total_val_size
            avg_hier_recall = total_hier_recall / total_val_size

            logger.info(
                f"Hierarchical evaluation: {dname}, {len(image_feats)} images, "
                f"{len(class_names)} classes \n[tie score: {avg_tie:.3f}, lca score: {avg_lca:.3f}, "
                f"jaccard similarity: {avg_jaccard:.3f}, hierarchical precision: {avg_hier_precision:.3f}, "
                f"hierarchical recall: {avg_hier_recall:.3f}] "
            )

        return results_dict