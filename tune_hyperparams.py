"""
This file takes care of tuning hyper-parameters and
fitting the final model
"""
import logging
import numpy as np

from sklearn.model_selection import GridSearchCV


def do_cross_validation(train_data, nn_class, param_grid, cv=3, fit_params={}):
    """
    Estimate variable importance, assumes we need to refit for each set of variable groups

    @param nn_class: a subclass of DecisionPredictionNNs (either the SimultaneousDensityDecisionNNs or SimultaneousIntervalDecisionNNs)
    @param dataset: Dataset
    @param param_grid: dictionary to CV over, contains all values for initializing NeuralNetworkAugMTL
                    (see docs for GridSearchCV from scikit)
    """
    # Pick best parameters via cross validation
    best_hyperparams, cv_results = _get_best_hyperparams(nn_class, param_grid, train_data, cv=cv, fit_params=fit_params)
    print("Done tuning hyperparams")
    logging.info("Best params %s", str(best_hyperparams))

    fitted_model = nn_class(**best_hyperparams)
    fitted_model.fit(train_data.x, train_data.y, **fit_params)
    return fitted_model, best_hyperparams, cv_results


def _get_best_hyperparams(model_cls, param_grid, dataset, cv=3, fit_params={}):
    """
    Runs cross-validation if needed
    @return best params chosen by CV in dict form, `cv_results_` attr from GridSearchCV
    """
    if np.all([len(v) == 1 for k, v in param_grid[0].items()]):
        # Don't run CV if there is nothing to tune
        return {k: v[0] for k, v in param_grid[0].items()}, None
    else:
        # grid search CV to get argmins
        # HACK: half the number of initializations for CV
        orig_num_inits = param_grid[0]["num_inits"][0]
        orig_support_sim_num = param_grid[0]["support_sim_num"][0]
        param_grid[0]["num_inits"][0] = max(orig_num_inits/2, 1)
        param_grid[0]["support_sim_num"][0] = int(orig_support_sim_num/3)
        grid_search_cv = GridSearchCV(
                model_cls(),
                param_grid=param_grid,
                cv=cv,
                #n_jobs=cv,
                refit=False,
                fit_params=fit_params)

        ### do cross validation
        grid_search_cv.fit(dataset.x, dataset.y)
        logging.info("Completed CV")
        logging.info(grid_search_cv.cv_results_)

        grid_search_cv.best_params_["num_inits"] = orig_num_inits
        grid_search_cv.best_params_["support_sim_num"] = orig_support_sim_num
        return grid_search_cv.best_params_, grid_search_cv.cv_results_
