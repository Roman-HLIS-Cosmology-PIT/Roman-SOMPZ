import os
from typing import Any

import qp

from rail.core.data import QPHandle, TableHandle
from rail.core.stage import RailStage
from rail.estimation.algos import naive_stack, point_est_hist, var_inf
from rail.utils.path_utils import RAILDIR

testdata = os.path.join(RAILDIR, "rail/examples_data/testdata/output_BPZ_lite.hdf5")
tomobins = os.path.join(RAILDIR, "rail/examples_data/testdata/output_tomo.hdf5")
DS = RailStage.data_store


def one_algo(
    key: str, summarizer_class: type[RailStage], summary_kwargs: dict[str, Any]
) -> qp.Ensemble:
    """
    A basic test of running an summaizer subclass
    Run summarize
    """
    DS.__class__.allow_overwrite = True
    DS.clear()
    test_data = DS.read_file("test_data", QPHandle, testdata)
    summarizer = summarizer_class.make_stage(name=key, **summary_kwargs)
    summary_ens = summarizer.summarize(test_data)
    os.remove(
        summarizer.get_output(summarizer.get_aliased_tag("output"), final_name=True)
    )
    os.remove(
        summarizer.get_output(summarizer.get_aliased_tag("single_NZ"), final_name=True)
    )
    return summary_ens


def one_mask_algo(
    key: str, summarizer_class: type[RailStage], summary_kwargs: dict[str, Any]
) -> list[qp.Ensemble]:
    """
    A basic test of running an summaizer subclass
    Run summarize
    """
    DS.__class__.allow_overwrite = True
    DS.clear()
    test_data = DS.read_file("test_data", QPHandle, testdata)
    tomo_bins = DS.read_file("tomo_bins", TableHandle, tomobins)

    summarizer = summarizer_class.make_stage(name=key, selected_bin=1, **summary_kwargs)
    summary_ens = summarizer.summarize(test_data, tomo_bins)
    os.remove(
        summarizer.get_output(summarizer.get_aliased_tag("output"), final_name=True)
    )
    os.remove(
        summarizer.get_output(summarizer.get_aliased_tag("single_NZ"), final_name=True)
    )

    summarizer_2 = summarizer_class.make_stage(name=f"{key}_2", **summary_kwargs)
    summary_2_ens = summarizer_2.summarize(test_data, None)
    os.remove(
        summarizer_2.get_output(summarizer_2.get_aliased_tag("output"), final_name=True)
    )
    os.remove(
        summarizer_2.get_output(
            summarizer_2.get_aliased_tag("single_NZ"), final_name=True
        )
    )

    return [summary_ens, summary_2_ens]


def test_naive_stack() -> None:
    """Basic end to end test for the Naive stack informer to estimator stages"""
    naive_stack_informer_stage = naive_stack.NaiveStackInformer.make_stage()
    naive_stack_informer_stage.inform()

    summary_config_dict: dict = {}
    summarizer_class = naive_stack.NaiveStackSummarizer
    _ = one_algo("NaiveStack", summarizer_class, summary_config_dict)
    os.remove(
        naive_stack_informer_stage.get_output(
            naive_stack_informer_stage.get_aliased_tag("model"), final_name=True
        )
    )


def test_point_estimate_hist() -> None:
    """Basic end to end test for the point estimate histogram informer to estimator
    stages
    """
    point_est_informer_stage = point_est_hist.PointEstHistInformer.make_stage()
    point_est_informer_stage.inform()

    summary_config_dict: dict = {}
    summarizer_class = point_est_hist.PointEstHistSummarizer
    _ = one_algo("PointEstimateHist", summarizer_class, summary_config_dict)
    os.remove(
        point_est_informer_stage.get_output(
            point_est_informer_stage.get_aliased_tag("model"), final_name=True
        )
    )


def test_var_inference_stack() -> None:
    """Basic end to end test for the var inference informer to estimator stages"""
    var_inf_informer_stage = var_inf.VarInfStackInformer.make_stage()
    var_inf_informer_stage.inform()

    summary_config_dict: dict = {}
    summarizer_class = var_inf.VarInfStackSummarizer
    _ = one_algo("VariationalInference", summarizer_class, summary_config_dict)
    os.remove(
        var_inf_informer_stage.get_output(
            var_inf_informer_stage.get_aliased_tag("model"), final_name=True
        )
    )


def test_naive_stack_masked() -> None:
    """Basic end to end test for the Naive stack informer to estimator stages"""
    summary_config_dict = dict(
        chunk_size=5,
    )
    summarizer_class = naive_stack.NaiveStackMaskedSummarizer
    _ = one_mask_algo("NaiveStack", summarizer_class, summary_config_dict)
    _ = one_algo("NaiveStack", summarizer_class, summary_config_dict)


def test_point_estimate_hist_masked() -> None:
    """Basic end to end test for the point estimate histogram informer to estimator
    stages
    """
    summary_config_dict = dict(
        chunk_size=5,
    )
    summarizer_class = point_est_hist.PointEstHistMaskedSummarizer
    _ = one_mask_algo("PointEstimateHist", summarizer_class, summary_config_dict)
    _ = one_algo("PointEstimateHist", summarizer_class, summary_config_dict)
