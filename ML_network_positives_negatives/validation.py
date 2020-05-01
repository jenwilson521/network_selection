from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate
import pandas as pd
import numpy as np

def plot_cm(tp,tn,fp,fn):
    cm = [["tp:"+str(tp), "fn:"+str(fn)], 
          ["fp:"+str(fp), "tn:"+str(tn)]]
    return cm


def nestedCrossValidation(estimator, grid, x, y, inner_split, outer_split, score_metrics, refit=True, shuffle=True, n_jobs=1, n_iter_search=None, ran_state=42):
    """
    Function that perform  nestesd cross validation. 
    If are provided the metrics to compute true positive, true negative, false positive and false negative (expressed as tp,tn,fp,fn),
    then the confusion matrix is returned into the nested cv results.
    
    :param estimator: scikit-learn estimator
    :param grid: [dict] parameter settings to test
    :param x: features
    :param y: targets
    :param inner_split: number of split into inner cross validation
    :param outer_split: number of split into inner cross validation
    :param shuffle: if True shuffle data before splitting
    """

    inner_cv = StratifiedKFold(n_splits=inner_split, shuffle=shuffle, random_state=ran_state)
    outer_cv = StratifiedKFold(n_splits=outer_split, shuffle=shuffle, random_state=ran_state)

    if n_iter_search is None:

        clf = GridSearchCV(estimator=estimator, param_grid=grid, cv=inner_cv, scoring=score_metrics, n_jobs=n_jobs, refit=refit)
    else:
        clf = RandomizedSearchCV(estimator=estimator, param_distributions=grid, n_iter=n_iter_search, cv=inner_cv, scoring=score_metrics, n_jobs=n_jobs, refit=refit, random_state=ran_state)

    nested_score = cross_validate(clf, scoring=score_metrics, X=x, y=y, cv=outer_cv, return_estimator=True, n_jobs=n_jobs)

    cm_elements_check = lambda n: sum([x in n for x in ["tp","tn","fp","fn"]]) == 0
    r = {"nested_score": {k:nested_score[k] for k in nested_score.keys() if "estimator" != k},
         "mean": [(n, nested_score["test_"+n].mean()) for n in score_metrics.keys() if cm_elements_check(n)],
         "std": [(n, nested_score["test_"+n].std()) for n in score_metrics.keys() if cm_elements_check(n)],
         "str": [n + " " + str(round(nested_score["test_"+n].mean(), 4)) + "+/-" + str(round(nested_score["test_"+n].std(), 4)) for n in score_metrics.keys() if cm_elements_check(n)]}
    
    if sum([not cm_elements_check(n) for n in score_metrics.keys()]) == 4: 
        r["str"] += ["cm"+str(i) + " " + str(plot_cm(nested_score["test_tp"][i], nested_score["test_tn"][i], nested_score["test_fp"][i], nested_score["test_fn"][i])) for i in range(outer_split)]
        
    if refit and sum([n in type(estimator).__name__  for n in ["DecisionTree", "RandomForest", "LogisticRegression"]]) > 0:
        contributions = []
        _search_cv = nested_score["estimator"]  # list of fitted GridSearchCV/RandomizedSearchCV

        for scv, (tr_ids, ts_ids) in zip(_search_cv, outer_cv.split(x, y)):
            best_estimator = scv.best_estimator_
            
            if type(best_estimator).__name__ == "LogisticRegression":
                contribution = best_estimator.coef_
            else:
                contribution = best_estimator.feature_importances_

                
            contributions.append(contribution)
        
        contr_mean = np.mean(np.asarray(contributions), axis=0)
        contr_std = np.std(contributions, axis=0)
        r["features_importance"] = list(zip(contr_mean, contr_std))
        if isinstance(x, pd.DataFrame):
            r["str"] += [col + ": "+ str(round(m, 4)) + " +/- " + str(round(s, 4)) for col, m, s in zip(x.columns, contr_mean, contr_std)]
    
    return r


def report(grid_scores, n_top=3, metrics_names="avg_per_class_accuracy"):
    """
    Report top n_top parameters settings, default n_top=3

    :param grid_scores: output from grid or random search
    :param n_top: how many to report, of top models
    :param metrics_names: name of the metrics to look at
    :return: top_params: [dict] top parameter settings found in
                  search
    """
    if len(grid_scores["mean_test_" + metrics_names[0]]) < n_top:
        n_top = len(grid_scores["mean_test_" + metrics_names[0]])

    top_scores = {"params": []}
    for mn in metrics_names:
        top_scores["mean_test_" + mn] = []
        top_scores["std_test_" + mn] = []

        if "mean_train_" + mn in grid_scores.keys():
            top_scores["mean_train_" + mn] = []
            top_scores["std_train_" + mn] = []

    rank = grid_scores["rank_test_" + metrics_names[0]].tolist()
    i = 1
    while n_top > 0:
        ii = [ind for ind, val in enumerate(rank) if val == i]
        if n_top < len(ii):
            for k in top_scores.keys():
                for ind in ii[:n_top]:
                    top_scores[k].append(grid_scores[k][ind])
            n_top = 0
        else:
            for k in top_scores.keys():
                for ind in ii:
                    top_scores[k].append(grid_scores[k][ind])
            n_top = n_top - len(ii)
        i = (i + len(ii))

    return top_scores


def run_gridsearch(X, y, clf, param_grid, score_metrics, refit=True, cv=5, n_jobs=3):
    """
    Run a grid search for best estimator parameters.
    :param X: features
    :param y: targets (classes)
    :param clf: scikit-learn classifier
    :param param_grid: [dict] parameter settings to test
    :param score_metrics: scoring metric of dict of scoring metrics
    :param refit: string denoting the scorer that would be used to find the best parameters for refitting the estimator at the end
    :param cv: fold of cross-validation, default 5
    :param n_jobs: number of jobs to run in parallel
    :return top_params: [dict] from report()
    """
    grid_search = GridSearchCV(clf,
                               scoring=score_metrics,
                               param_grid=param_grid,
                               refit=refit,
                               return_train_score=True,
                               cv=cv, n_jobs=n_jobs)

    grid_search.fit(X, y)
    k = refit if isinstance(refit, str) else "score"
    mn = [k]
    if isinstance(refit, str):
        mn += [n for n in score_metrics.keys() if n != refit]
    return grid_search, report(grid_search.cv_results_, len(grid_search.cv_results_["mean_test_" + k]), mn)


def run_randomsearch(X, y, clf, param_dist, score_metrics, cv=5, refit=True, n_iter_search=20, n_jobs=3, ran_state=42):
    """
    Run a random search for best Decision Tree parameters.
    :param X: features
    :param y: targets (classes)
    :param clf: scikit-learn classifier
    :param param_dist: [dict] list, distributions of parameters to sample
    :param score_metrics: scoring metric of dict of scoring metrics
    :param refit: string denoting the scorer that would be used to find the best parameters for refitting the estimator at the end
    :param cv: fold of cross-validation, default 5
    :param n_iter_search: number of random parameter sets to try, default 20
    :param n_jobs: number of jobs to run in parallel
    :return top_params: [dict] from report()
    """
    random_search = RandomizedSearchCV(clf,
                                       scoring=score_metrics,
                                       param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       cv=cv, n_jobs=n_jobs,
                                       refit=refit,
                                       return_train_score=True,
                                       random_state=ran_state)

    random_search.fit(X, y)
    k = refit if isinstance(refit, str) else "score"
    mn = [k]
    if isinstance(refit, str):
        mn += [n for n in score_metrics.keys() if n != refit]
    return random_search, report(random_search.cv_results_, len(random_search.cv_results_["mean_test_" + k]), mn)
