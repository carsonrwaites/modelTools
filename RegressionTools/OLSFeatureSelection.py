import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import statsmodels.api as sm

## NOTES: This is not an exhaustive feature selection,
##      i.e., it doesn't iterate through all 2^p possible feature combinations.
##      Instead, it always uses a stepwise forward or backward algorithm,
##      keeping the dropped or retained variables additive through steps.
##      This is a limitation, but computationally much less intense.
##      Also, 'loud' arguments determine whether intermediate steps/graphs are produced.
## FUTURE: Add backwards CV selection?

def p_forward(X, y, thresh=0.05, loud=False):
    """
    A step-forward feature selection, choosing the features that lower RSS the most.
    New variables are subject to the p-value threshold set in the argument.
    In current construction, thresh=1 eliminates p-value constraint.
    This is not recommended, I have covered this style of selection in the CV function.
    """
    rem_features = list(X.columns)
    keep = []
    while len(rem_features) > 0:
        results = {}
        for rem_feat in rem_features:
            test_feats = keep + [rem_feat]
            df = sm.add_constant(X[test_feats])

            model = sm.OLS(y, df).fit()
            pval = model.pvalues[rem_feat]
            if pval <= thresh:
                RSS = np.square(model.resid).sum()
                results[rem_feat] = (RSS, pval)

        # Stop if no remaining feature meets p-value requirement
        if len(results) == 0:
            if loud:
                print("No remaining features meet p-value threshold.")
            break

        # Select feature that minimizes RSS
        best_feat = min(results, key=lambda k: results[k][0])

        keep.append(best_feat)
        rem_features.remove(best_feat)

        if loud:
            print(f"Best feature: {best_feat}, {results[best_feat]}")
    if loud:
        print(f"Final features: {keep}")
        print(f"Unused features: {rem_features}")
    return keep


def p_backward(X, y, thresh=0.05, loud=False):
    """
    This is a backward selection method, always dropping the feature with the greatest p-value.
    Here, a feature won't be dropped if its p-value is below the threshold in the argument.
    This value must be >0, otherwise all features will be dropped.
    """
    rem_features = list(X.columns)
    while True:
        model = sm.OLS(y, sm.add_constant(X[rem_features])).fit()
        pvals = dict(model.pvalues.drop(['const']))
        if max(pvals.values()) >= thresh:
            remove_feature = max(pvals, key=lambda k: pvals[k])
            rem_features.remove(remove_feature)
            if loud:
                print(f"{remove_feature} removed; p-value {pvals[remove_feature]}")
        else:
            if loud:
                print(f"All remaining features meet p-value threshold.")
            break
    if loud:
        print(f"Final features: {rem_features}")
    return rem_features

def cv_forward(X, y, max_d=None, test_size=0.2, n_splits=5, random_state=0, loud=False):
    """
    This is a forward-stepping cross-validation selection.
    This algorithm selects which feature increases performance the most per the CV.
    Features are retained through steps, so only one is considered at a time.
    Finally, it also measures performance on a hold-out test set, and plots both per value of d.
    """
    if max_d is None:
        max_d = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    rem_features = list(X.columns)
    keep = []
    results = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for d in range(1, max_d + 1):
        cv_scores = {}
        for feat in rem_features:
            feats = keep + [feat]
            Xd = X_train[feats].values
            fold_mse = []

            for train_idx, val_idx in kf.split(Xd):
                X_tr, X_val = Xd[train_idx], Xd[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                X_tr = sm.add_constant(X_tr)
                X_val = sm.add_constant(X_val)

                model = sm.OLS(y_tr, X_tr).fit()
                preds = model.predict(X_val)
                fold_mse.append(mean_squared_error(y_val, preds))

            cv_scores[feat] = np.mean(fold_mse)

        # Choose feature with the best CV performance
        best_feat = min(cv_scores, key=cv_scores.get)
        keep.append(best_feat)
        rem_features.remove(best_feat)

        # Refit model on full training set
        X_train_d = sm.add_constant(X_train[keep])
        X_test_d = sm.add_constant(X_test[keep])

        final_model = sm.OLS(y_train, X_train_d).fit()

        cv_score = cv_scores[best_feat]

        test_preds = final_model.predict(X_test_d)
        test_mse = mean_squared_error(y_test, test_preds)

        results[d] = {
            "features": keep.copy(),
            "cv_mse": cv_score,
            "test_mse": test_mse
        }

    # Plotting CV and test results
    if loud:
        ds = list(results.keys())
        cv_vals = [results[d]["cv_mse"] for d in ds]
        test_vals = [results[d]["test_mse"] for d in ds]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(ds, cv_vals, marker="o")
        ax1.set_xlabel("Number of Features (d)")
        ax1.set_ylabel("CV MSE")
        ax1.set_title("Cross-Validation Performance")

        ax2.plot(ds, test_vals, marker="o")
        ax2.set_xlabel("Number of Features (d)")
        ax2.set_ylabel("Test MSE")
        ax2.set_title("Test Performance")

        plt.show()

    return results

# Example usage. Here I'm pulling a dataset from UC Irvine's data packages.
if __name__=="__main__":
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=165)
    X = dataset.data.features
    y = dataset.data.targets

    keep_forward = sorted(p_forward(X, y, thresh=0.05, loud=False))
    keep_backward = sorted(p_backward(X, y, thresh=0.05, loud=False))

    print(f"Forward search:  {keep_forward}")
    print(f"Backward search: {keep_backward}")

    results = cv_forward(X, y, loud=True)
    print(results)
