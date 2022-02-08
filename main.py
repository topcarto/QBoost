import numpy
import sys
from tabulate import tabulate

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

from dwave.system import LeapHybridSampler
import dimod

def qboost_lambda_sweep(X, y, lambda_vals, val_fraction=0.4, verbose=True, **kwargs):
    """Run QBoost using a series of lambda values and check accuracy against a validation set.

    Args:
        X (array):
            2D array of feature vectors.
        y (array):
            1D array of class labels (+/- 1).
        lambda_vals (array):
            Array of values for regularization parameter, lambda.
        val_fraction (float):
            Fraction of given data to set aside for validation.
        verbose (bool):
            Print out diagnostic information to screen.
        kwargs:
            Passed to QBoost.__init__.

    Returns:
        QBoostClassifier:
            QBoost instance with best validation score.
        lambda:
            Lambda value corresponding to the best validation score.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_fraction)

    best_score = -1
    best_lambda = None
    best_clf = None

    if verbose:
        print('{:7} {} {}:'.format('lambda', 'n_features', 'score'))

    for lam in lambda_vals:
        qb = QBoostClassifier(X_train, y_train, lam, **kwargs)
        score = qb.score(X_val, y_val)
        if verbose:
            print('{:<7.4f} {:<10} {:<6.3f}'.format(
                lam, len(qb.get_selected_features()), score))
        if score > best_score:
            best_score = score
            best_clf = qb
            best_lambda = lam

    return best_clf, lam

def _build_H(classifiers, X, output_scale):
    """Construct matrix of weak classifier predictions on given set of input vectors."""
    H = numpy.array([clf.predict(X) for clf in classifiers], dtype=float).T

    # Rescale H
    H *= output_scale

    return H

def _minimize_squared_loss_binary(H, y, lam):
    """Minimize squared loss using binary weight variables."""
    bqm = _build_bqm(H, y, lam)

    sampler = LeapHybridSampler()
    results = sampler.sample(bqm)
    sample = results.first.sample
    weights = numpy.array(list(results.first.sample.values()))
    energy = results.first.energy

    selected_item_indices = []
    for varname, value in sample.items():
        # For each "x" variable, check whether its value is set, which
        # indicates that the corresponding item is included in the
        # knapsack
            # The index into the weight array is retrieved from the
            # variable name
        if(sample[varname] == 1):
            selected_item_indices.append(int(varname))

    return sorted(selected_item_indices), sample, weights, energy

def _build_bqm(H, y, lam):
    """Build BQM.

    Args:
        H (array):
            2D array of weak classifier predictions.  Each row is a
            sample point, each column is a classifier.
        y (array):
            Outputs
        lam (float):
            Coefficient that controls strength of regularization term
            (larger values encourage decreased model complexity).
    """
    n_samples = numpy.size(H, 0)
    n_classifiers = numpy.size(H, 1)

    # samples_factor is a factor that appears in front of the squared
    # loss term in the objective.  In theory, it does not affect the
    # problem solution, but it does affect the relative weighting of
    # the loss and regularization terms, which is otherwise absorbed
    # into the lambda parameter.

    # Using an average seems to be more intuitive, otherwise, lambda
    # is sample-size dependent.
    samples_factor = 1.0 / n_samples

    bqm = dimod.BQM('BINARY')
    bqm.offset = samples_factor * n_samples

    for i in range(n_classifiers):
        # Note: the last term with h_i^2 is part of the first term in
        # Eq. (12) of Neven et al. (2008), where i=j.
        bqm.add_variable(i, lam - 2.0 * samples_factor *
                         numpy.dot(H[:, i], y) + samples_factor * numpy.dot(H[:, i], H[:, i]))

    for i in range(n_classifiers):
        for j in range(i+1, n_classifiers):
            # Relative to Eq. (12) from Neven et al. (2008), the
            # factor of 2 appears here because each term appears twice
            # in a sum over all i,j.
            bqm.add_interaction(
                i, j, 2.0 * samples_factor * numpy.dot(H[:, i], H[:, j]))

    return bqm

class DecisionStumpClassifier:
    """Decision tree classifier that operates on a single feature with a single splitting rule.

    The index of the feature used in the decision rule is stored
    relative to the original data frame.
    """

    def __init__(self, X, y, feature_index):
        """Initialize and fit the classifier.

        Args:
            X (array): 
                2D array of feature vectors.  Note that the array
                contains all features, while the weak classifier
                itself uses only a single feature.
            y (array):
                1D array of class labels, as ints.  Labels should be
                +/- 1.
            feature_index (int):
                Index for the feature used by the weak classifier,
                relative to the overall data frame.
        """
        self.i = feature_index

        self.clf = DecisionTreeClassifier(max_depth=1)
        self.clf.fit(X[:, [feature_index]], y)

    def predict(self, X):
        """Predict class.

        Args:
            X (array):
                2D array of feature vectors.  Note that the array
                contains all features, while the weak classifier
                itself will make a prediction based only a single
                feature.

        Returns:
            Array of class labels.
        """
        return self.clf.predict(X[:, [self.i]])

class QBoostClassifier:
    def __init__(self, X, y, lam, weak_clf_scale=None, drop_unused=True):

        if not all(numpy.isin(y, [-1, 1])):
            raise ValueError("Class labels should be +/- 1")

        num_features = numpy.size(X, 1)

        if weak_clf_scale is None:
            weak_clf_scale = 1 / num_features

        wclf_candidates = [DecisionStumpClassifier(
            X, y, i) for i in range(num_features)]

        H = _build_H(wclf_candidates, X, weak_clf_scale)

        # For reference, store individual weak classifier scores.
        # Note: we don't check equality h==y here because H might be rescaled.
        self.weak_scores = numpy.array([numpy.mean(numpy.sign(h) * y > 0) for h in H.T])

        self.selected_items, self.sample, self.w, self.energy = _minimize_squared_loss_binary(H, y, lam)
        self.weak_classifiers = wclf_candidates
        self.weak_clf_scale = weak_clf_scale

        self.fit_offset(X)

        # Save candidates so we can provide a baseline accuracy report.
        self._wclf_candidates = wclf_candidates

    def fit_offset(self, X):
        """Fit offset value based on class-balanced feature vectors.

        Currently, this assumes that the feature vectors in X
        correspond to an even split between both classes.
        """
        self.offset = 0.0
        # Todo: review whether it would be appropriate to subtract
        # mean(y) here to account for unbalanced classes.
        self.offset = numpy.mean(self.predict(X))

    def get_selected_features(self):
        selected_item_indices = []
        for varname, value in self.sample.items():
            # For each "x" variable, check whether its value is set, which
            # indicates that the corresponding item is included in the
            # knapsack
                # The index into the weight array is retrieved from the
                # variable name
            if(self.sample[varname] == 1):
                selected_item_indices.append(int(varname))

        return selected_item_indices

    def score(self, X, y):
        """Compute accuracy score on given data."""
        if sum(self.w) == 0:
            # Avoid difficulties that occur with handling this below
            return 0.0
        return accuracy_score(y, self.predict_class(X))

    def predict(self, X):
        """Compute ensemble prediction.

        Note that this function returns the numerical value of the
        ensemble predictor, not the class label.  The predicted class
        is sign(predict()).
        """
        H = _build_H(self.weak_classifiers, X, self.weak_clf_scale)

        # If we've already filtered out those with w=0 and we are only
        # using binary weights, this is just a sum
        preds = numpy.dot(H, self.w)
        return preds - self.offset

    def predict_class(self, X):
        """Compute ensemble prediction of class label."""
        preds = self.predict(X)

        # Add a small perturbation to any predictions that are exactly
        # 0, because these will not count towards either class when
        # passed through the sign function.  Such zero predictions can
        # happen when the weak classifiers exactly balance each other
        # out.
        preds[preds == 0] = 1e-9

        return numpy.sign(preds)

    def report_baseline(self, X, y):
        """Report accuracy of weak classifiers.

        This provides context for interpreting the performance of the boosted
        classifier.
        """
        scores = numpy.array([accuracy_score(y, clf.predict(X))
                           for clf in self._wclf_candidates])
        data = [[len(scores), scores.min(), scores.mean(), scores.max(), scores.std()]]
        headers = ['count', 'min', 'mean', 'max', 'std']

        print('Accuracy of weak classifiers (score on test set):')
        print(tabulate(data, headers=headers, floatfmt='.3f'))

def make_blob_data(n_samples=100, n_features=5, n_informative=2, delta=1):
    """Generate sample data based on isotropic Gaussians with a specified number of class-informative features.

    Args:
        n_samples (int):
            Number of samples.
        n_features (int):
            Number of features.
        n_informative (int):
            Number of informative features.
        delta (float):
            Difference in mean values of the informative features
            between the two classes.  (Note that all features have a
            standard deviation of 1).

    Returns:
        X (array of shape (n_samples, n_features))
            Feature vectors.
        y (array of shape (n_samples,):
            Class labels with values of +/- 1.
    """
    if n_informative > n_features:
        raise ValueError("n_informative must be less than or equal to n_features")

    # Set up the centers so that only n_informative features have a
    # different mean for the two classes.
    class0_centers = numpy.zeros(n_features)
    class1_centers = numpy.zeros(n_features)
    class1_centers[:n_informative] = delta
    
    centers = numpy.vstack((class0_centers, class1_centers))
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers)

    # Convert class labels to +/- 1
    y = y * 2 - 1

    return X, y

if __name__ == '__main__':
    n_samples = int(sys.argv[1])
    n_features = int(sys.argv[2])
    n_informative = int(sys.argv[3])
    lam = float(sys.argv[4])

    X, y = make_blob_data(
        n_samples=n_samples, n_features=n_features, n_informative=n_informative)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4)


    normalized_lambdas = numpy.linspace(0.0, 0.5, 10)
    lambdas = normalized_lambdas / n_features
    print('Performing cross-validation using {} values of lambda, this may take several minutes...'.format(len(lambdas)))
    qboost, lam = qboost_lambda_sweep(
        X_train, y_train, lambdas)
    qboost.report_baseline(X_test, y_test)

    print('Informative features:', list(range(n_informative)))
    print('Selected features:', qboost.get_selected_features())

    print('Score on test set: {:.3f}'.format(qboost.score(X_test, y_test)))
