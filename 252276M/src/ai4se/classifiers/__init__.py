"""Classifiers sub-package."""


from .base import Classifier, ClassifierFactory, train_per_repo, train_pooled
from .classical import TfidfClassifier, make_classical, make_tuned_linear_svm, make_tuned_logreg
from .ensemble import SoftVotingEnsemble, make_ensemble
from .floors import KeywordRulesClassifier, MajorityClassifier, ScratchNaiveBayesClassifier, StratifiedRandomClassifier
from .frozen import FrozenEncoderClassifier, make_frozen_minilm, make_frozen_mpnet
from .neural import FeedForwardClassifier, TextCnnClassifier, make_ffnn, make_text_cnn
from .setfit_model import SetFitClassifier, make_setfit_minilm_matched, make_setfit_minilm_repro, make_setfit_mpnet


__all__ = [
    "Classifier", "ClassifierFactory", "train_per_repo", "train_pooled",
    "MajorityClassifier", "StratifiedRandomClassifier", "KeywordRulesClassifier", "ScratchNaiveBayesClassifier",
    "TfidfClassifier", "make_classical", "make_tuned_logreg", "make_tuned_linear_svm",
    "FeedForwardClassifier", "TextCnnClassifier", "make_ffnn", "make_text_cnn",
    "FrozenEncoderClassifier", "make_frozen_minilm", "make_frozen_mpnet",
    "SetFitClassifier", "make_setfit_mpnet", "make_setfit_minilm_matched", "make_setfit_minilm_repro",
    "SoftVotingEnsemble", "make_ensemble",
]
