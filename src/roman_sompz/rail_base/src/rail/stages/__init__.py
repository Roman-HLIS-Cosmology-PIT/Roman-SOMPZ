import rail
from rail.core import RailEnv
from rail.core.stage import RailStage
from rail.creation.degrader import Degrader
from rail.creation.degraders.addRandom import AddColumnOfRandom
from rail.creation.degraders.quantityCut import QuantityCut
from rail.creation.engine import Creator, Modeler, PosteriorCalculator
from rail.creation.noisifier import Noisifier
from rail.creation.selector import Selector
from rail.estimation.algos.equal_count import EqualCountClassifier
from rail.estimation.algos.naive_stack import (NaiveStackInformer,
                                               NaiveStackSummarizer)
from rail.estimation.algos.point_est_hist import (PointEstHistInformer,
                                                  PointEstHistSummarizer)
from rail.estimation.algos.random_gauss import (RandomGaussEstimator,
                                                RandomGaussInformer)
from rail.estimation.algos.train_z import TrainZEstimator, TrainZInformer
from rail.estimation.algos.true_nz import TrueNZHistogrammer
from rail.estimation.algos.uniform_binning import UniformBinningClassifier
from rail.estimation.algos.var_inf import (VarInfStackInformer,
                                           VarInfStackSummarizer)
from rail.estimation.classifier import CatClassifier, PZClassifier
from rail.estimation.estimator import CatEstimator, PzEstimator
from rail.estimation.informer import CatInformer, PzInformer

from rail.estimation.summarizer import (CatSummarizer, PZSummarizer,
                                        SZPZSummarizer)
from rail.evaluation.dist_to_dist_evaluator import DistToDistEvaluator
from rail.evaluation.dist_to_point_evaluator import DistToPointEvaluator
from rail.evaluation.evaluator import Evaluator, OldEvaluator
from rail.evaluation.point_to_point_evaluator import PointToPointEvaluator
from rail.evaluation.single_evaluator import SingleEvaluator
from rail.tools.table_tools import ColumnMapper, RowSelector, TableConverter

__all__ = [
    "CatEstimator",
    "CatClassifier",
    "PZClassifier",
    "CatSummarizer",
    "PZSummarizer",
    "SZPZSummarizer",
    "NaiveStackInformer",
    "NaiveStackSummarizer",
    "RandomGaussInformer",
    "RandomGaussEstimator",
    "PointEstHistInformer",
    "PointEstHistSummarizer",
    "TrainZInformer",
    "TrainZEstimator",
    "VarInfStackInformer",
    "VarInfStackSummarizer",
    "UniformBinningClassifier",
    "EqualCountClassifier",
    "TrueNZHistogrammer",
    "Degrader",
    "AddColumnOfRandom",
    "QuantityCut",
    "Modeler",
    "Creator",
    "PosteriorCalculator",
    "Noisifier",
    "Selector",
    "DistToDistEvaluator",
    "DistToPointEvaluator",
    "Evaluator",
    "OldEvaluator",
    "PointToPointEvaluator",
    "SingleEvaluator",
    "ColumnMapper",
    "RowSelector",
    "TableConverter",
]


def import_and_attach_all() -> None:
    """Import all the packages in the rail ecosystem and attach them to this module"""
    RailEnv.import_all_packages()
    RailEnv.attach_stages(rail.stages)
    for xx in RailStage.pipeline_stages:
        rail.stages.__all__.append(xx)
