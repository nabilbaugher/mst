import pickle
import numpy as np
import pprint
import matplotlib.pyplot as plt
import random

import loglikes
import models

pp = pprint.PrettyPrinter(compact=False, width=90)

with open(f'__experiment_1/parsed_data/tree.pickle', 'rb') as handle:
    TREE = pickle.load(handle)

with open(f'__experiment_1/parsed_data/subject_decisions.pickle', 'rb') as handle:
    # {sid: {world: {'nodes': [], 'path': []}}}
    DECISIONS = pickle.load(handle)


def split_train_test_kfold(decisions_list, k=4):
    """ k = 4 """

    n = len(decisions_list) // k  # length of each split
    splits = [decisions_list[i * n: (i + 1) * n] for i in range(k)]

    for i in range(k):
        train = sum([splits[j] for j in range(k) if j != i], [])
        test = splits[i]

        yield train, test


def split_train_test_rand(decisions_list, k=4):
    """ 75% train, 25% test """

    for _ in range(k):
        test = random.sample(decisions_list, k=len(decisions_list) // k)
        train = [d for d in decisions_list if d not in test]

        yield train, test


def model_preference():
    """
    return model preference summary
    {model_name: number of subjects that prefer this model}
    """

    model2parameters = {
        'Expected_Utility': [(tau, 1, 1) for tau in models.TAUS],
        'Discounted_Utility': [(round(tau, 3), round(gamma, 3), 1) for tau in models.TAUS for gamma in models.GAMMAS],
        'Probability_Weighted_Utility': [(round(tau, 3), 1, round(beta, 3)) for tau in models.TAUS for beta in
                                         models.BETAS]
    }

    model_preference = {}  # {model_name: number of subjects that prefer this model}

    for sid in DECISIONS:
        decisions_list = []
        max_avg_loglike = float('-inf')

        # collect all decisions made by subject sid
        for world in DECISIONS[sid]:
            decisions_list.extend((world, nid) for nid in DECISIONS[sid][world]['nodes'])

        for model_name, parameters in model2parameters.items():

            avg_test_loglike, k = 0, 4

            for train, test in split_train_test_rand(decisions_list, k):
                max_loglike, mle_params = loglikes.mle(parameters, model_name, train)
                avg_test_loglike += loglikes.loglike(mle_params, model_name, test) / k

            if avg_test_loglike > max_avg_loglike:
                max_avg_loglike = avg_test_loglike
                best_model = model_name

        model_preference[best_model] = model_preference.get(best_model, 0) + 1

    pp.pprint(model_preference)
    return model_preference


if __name__ == "__main__":
    model_preference()