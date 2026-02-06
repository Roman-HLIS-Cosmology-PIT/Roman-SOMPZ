"""
A classifier that uses pz point estimate to assign
tomographic bins with uniform binning. 
"""

import numpy as np
from ceci.config import StageParameter as Param

from rail.core.data import Hdf5Handle
from rail.estimation.classifier import PZClassifier


class EqualCountClassifier(PZClassifier):
    """Classifier that simply assign tomographic
    bins based on point estimate according to SRD"""

    name = "EqualCountClassifier"
    config_options = PZClassifier.config_options.copy()
    config_options.update(
        id_name=Param(
            str,
            "",
            msg="Column name for the object ID in the input data, if empty the row index is used as the ID.",
        ),
        point_estimate=Param(str, "zmode", msg="Which point estimate to use"),
        zmin=Param(float, 0.0, msg="Minimum redshift of the sample"),
        zmax=Param(float, 3.0, msg="Maximum redshift of the sample"),
        nbins=Param(int, 5, msg="Number of tomographic bins"),
        no_assign=Param(int, -99, msg="Value for no assignment flag"),
    )
    outputs = [("output", Hdf5Handle)]

    def run(self) -> None:
        test_data = self.get_data("input")
        npdf = test_data.npdf

        try:
            zb = np.squeeze(test_data.ancil[self.config.point_estimate])
        except KeyError as msg:
            raise KeyError(
                f"{self.config.point_estimate} is not contained in the data ancil, "
                "you will need to compute it explicitly."
            ) from msg

        # tomographic bins with equal number density
        sortind = np.argsort(zb)
        cum = np.arange(1, (len(zb) + 1))
        bin_index = np.zeros(len(zb))
        for ii in range(self.config.nbins):
            perc1 = ii / self.config.nbins
            perc2 = (ii + 1) / self.config.nbins
            ind = (cum / cum[-1] > perc1) & (cum / cum[-1] <= perc2)
            useind = sortind[ind]
            bin_index[useind] = int(ii + 1)

        if self.config.id_name != "":
            # below is commented out and replaced by a redundant line
            # because the data doesn't have ID yet
            # obj_id = test_data[self.config.id_name]
            obj_id = np.arange(npdf)
        elif self.config.id_name == "":
            # ID set to row index
            obj_id = np.arange(npdf)
            self.config.id_name = "row_index"

        class_id = {self.config.id_name: obj_id, "class_id": bin_index}
        self.add_data("output", class_id)
