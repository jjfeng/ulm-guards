import numpy as np
from numpy import ndarray

class Dataset:
    """
    Stores data
    """
    def __init__(self,
            x: ndarray = None,
            y: ndarray = None,
            true_pdf: ndarray = None,
            group_id: ndarray = None,
            num_classes: int = None):
        """
        @param x: np array of covariates (each row is observation)
        @param y: column vector (N X 1) of responses
        @param true_pdf: the true pdf of Y|X
        @param group_id: vector (N) that specifies which group this
                    observation comes from (We need this to deal with correlated
                    observations. We will use this to split observations
                    by group, as well as later sampling one observation per group).
                    If not specified, we assume observations are completely independent.
        @param num_classes: number of classes, if a multinomial. otherwise this is None
        """
        self.x = x
        self.num_p = x.shape[1]
        self.y = y
        self.num_classes = num_classes
        self.true_pdf = true_pdf
        if group_id is None:
            # Suppose each observation is different group
            self.group_id = np.arange(0, x.shape[0])
        else:
            self.group_id = group_id

    @property
    def num_obs(self):
        return self.x.shape[0]

    def subset(self, idxs: ndarray):
        """
        @return a subset of the data using observations with row num = `idxs`
        """
        return Dataset(
            self.x[idxs,:],
            self.y[idxs,:],
            true_pdf=self.true_pdf[idxs,:] if self.true_pdf is not None else None,
            group_id=self.group_id[idxs],
            num_classes=self.num_classes)
