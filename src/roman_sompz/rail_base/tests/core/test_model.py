import os
import pickle
from typing import Any

import numpy as np
import pytest

from rail.core.model import Model
from rail.estimation.algos.train_z import trainZmodel


def pickle_dump(obj: Any, path: str) -> None:
    with open(path, "wb") as fout:
        pickle.dump(obj=obj, file=fout, protocol=pickle.HIGHEST_PROTOCOL)


def test_model() -> None:
    array_data = np.array([None])
    dict_data = dict(a=5, b=6)
    train_z_data = trainZmodel(np.linspace(0.0, 3.0, 301), np.ones(301), 1.5)
    model = Model(array_data, "dummy", 0, dict(alice="bob"))
    _model_2 = Model(array_data, "dummy", 0)

    pickle_dump(array_data, "array_raw.pickle")
    _read_array_data = Model.read("array_raw.pickle", "dummy", 0, dict(alice="bob"))
    _read_array_data_2 = Model.read("array_raw.pickle", "dummy", 0)

    Model.wrap("array_raw.pickle", "array_wrap.pickle", "dummy", 0, dict(alice="bob"))

    Model.dump(array_data, "array_data.pickle", "dummy", 0)
    Model.dump(dict_data, "dict_data.pickle", "dummy", 0, dict(alice="bob"))
    Model.dump(train_z_data, "train_z_data.pickle", "dummy", 0, dict(alice="bob"))
    Model.dump(model, "model_data.pickle")

    check_array_data = Model.read("array_data.pickle")
    _check_dict_data = Model.read("dict_data.pickle")
    _check_train_z_data = Model.read("train_z_data.pickle")
    _check_model_data = Model.read("model_data.pickle")

    Model.validate(check_array_data, "dummy", 0)
    with pytest.raises(TypeError):
        Model.validate(check_array_data, "dummies", 0)

    with pytest.raises(TypeError):
        Model.validate(check_array_data, "dummy", 1)

    os.remove("array_data.pickle")
    os.remove("array_raw.pickle")
    os.remove("array_wrap.pickle")
    os.remove("dict_data.pickle")
    os.remove("model_data.pickle")
    os.remove("train_z_data.pickle")
