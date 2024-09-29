
import torch
import numpy as np 
import scipy

import matplotlib.pyplot as plt 

# code adapted from https://github.com/fregu856/evaluating_bdl/blob/master/depthCompletion/ensembling_eval_auce.py

def auce(mean_values, sigma_values, target_values):
    
    auce_dict = {}
    num_predictions_with_GT = float(np.prod(target_values.shape))
    
    coverage_values = []
    avg_length_values = []
    alphas = list(np.arange(start=0.01, stop=1.0, step=0.01)) # ([0.01, 0.02, ..., 0.99], 99 elements)
    for step, alpha in enumerate(alphas):
        #print ("alpha: %d/%d" % (step+1, len(alphas)))
        
        lower_values = mean_values - scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_values # (shape: (num_predictions_with_GT, ))
        upper_values = mean_values + scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_values # (shape: (num_predictions_with_GT, ))

        coverage = np.count_nonzero(np.logical_and(target_values >= lower_values, target_values <= upper_values)) / num_predictions_with_GT
        coverage_values.append(coverage)
        
        avg_length = np.mean(upper_values - lower_values)
        avg_length_values.append(avg_length)
        
    auc_length = np.trapz(y=avg_length_values, x=alphas)
    # print ("AUCE - Length: %g" % auc_length)

    coverage_error_values =  np.array(coverage_values) - (1.0 - np.array(alphas))
    
    abs_coverage_error_values = np.abs(coverage_error_values)
    
    neg_coverage_error_values = (np.abs(coverage_error_values) - coverage_error_values) / 2.0
    
    auc_error = np.trapz(y=abs_coverage_error_values, x=alphas)
    # print ("AUCE - Empirical coverage absolute error: %g" % auc_error)
    
    auc_neg_error = np.trapz(y=neg_coverage_error_values, x=alphas)
    # print ("AUCE - Empirical coverage negative error: %g" % auc_neg_error)
    
    # store results in dict
    auce_dict["coverage_values"] = np.array(coverage_values)
    auce_dict["avg_length_values"] = np.array(avg_length_values)
    auce_dict["coverage_error_values"] = np.array(coverage_error_values)
    auce_dict["abs_coverage_error_values"] = abs_coverage_error_values
    auce_dict["neg_coverage_error_values"] = neg_coverage_error_values

    auce_dict["auc_abs_error_values"] = auc_error
    auce_dict["auc_length_values"] = auc_length
    auce_dict["auc_neg_error_values"] = auc_neg_error
    
    # print()
    return auce_dict
        


def plot_auce_curves(coverage_values, 
                     avg_length_values, 
                     coverage_error_values, 
                     abs_coverage_error_values, 
                     neg_coverage_error_values,
                     save_dir="./imgs",
                     output="rgb"):

    alphas = list(np.arange(start=0.01, stop=1.0, step=0.01)) # ([0.01, 0.02, ..., 0.99], 99 elements)

    plt.figure(1)
    plt.plot([0.0, 1.0], [0.0, 1.0], "k:", label="Perfect")
    # plt.plot(alphas, coverage_values)
    plt.plot(alphas, np.flip(coverage_values, 0))
    plt.legend()
    plt.ylabel("Empirical coverage")
    plt.xlabel("p")
    plt.title("Prediction intervals - Empirical coverage")
    plt.savefig("%s/%s_empirical_coverage.png" % (save_dir, output))
    plt.close(1)

    plt.figure(1)
    # plt.plot(alphas, avg_length_values)
    plt.plot(alphas, np.flip(avg_length_values, 0))
    # plt.legend()
    plt.ylabel("Average interval length [m]")
    plt.xlabel("p")
    avg_length_ylim = plt.ylim()
    plt.title("Prediction intervals - Average interval length")
    plt.savefig("%s/%s_length.png" % (save_dir, output))
    plt.close(1)

    plt.figure(1)
    plt.plot([0.0, 1.0], [0.0, 0.0], "k:", label="Perfect")
    # plt.plot(alphas, coverage_error_values)
    plt.plot(alphas, np.flip(coverage_error_values, 0))
    plt.legend()
    plt.ylabel("Empirical coverage error")
    plt.xlabel("p")
    coverage_error_ylim = plt.ylim()
    plt.title("Prediction intervals - Empirical coverage error")
    plt.savefig("%s/%s_empirical_coverage_error.png" % (save_dir, output))
    plt.close(1)

    plt.figure(1)
    plt.plot([0.0, 1.0], [0.0, 0.0], "k:", label="Perfect")
    # plt.plot(alphas, abs_coverage_error_values)
    plt.plot(alphas, np.flip(abs_coverage_error_values, 0))
    plt.legend()
    plt.ylabel("Empirical coverage absolute error")
    plt.xlabel("p")
    abs_coverage_error_ylim = plt.ylim()
    plt.title("Prediction intervals - Empirical coverage absolute error")
    plt.savefig("%s/%s_empirical_coverage_absolute_error.png" % (save_dir, output))
    plt.close(1)

    plt.figure(1)
    plt.plot([0.0, 1.0], [0.0, 0.0], "k:", label="Perfect")
    # plt.plot(alphas, neg_coverage_error_values)
    plt.plot(alphas, np.flip(neg_coverage_error_values, 0))
    plt.legend()
    plt.ylabel("Empirical coverage negative error")
    plt.xlabel("p")
    neg_coverage_error_ylim = plt.ylim()
    plt.title("Prediction intervals - Empirical coverage negative error")
    plt.savefig("%s/%s_empirical_coverage_negative_error.png" % (save_dir, output))
    plt.close(1)
    
    ## save values
    np.save("%s/auce_%s_alphas.npy" % (save_dir, output), 
            alphas)
    np.save("%s/auce_%s_empirical_coverage.npy" % (save_dir, output), 
            coverage_values)
    np.save("%s/auce_%s_avg_length.npy" % (save_dir, output), 
            avg_length_values)
    np.save("%s/auce_%s_empirical_coverage_error.npy" % (save_dir, output), 
            coverage_error_values)
    np.save("%s/auce_%s_empirical_coverage_absolute_error.npy" % (save_dir, output), 
            abs_coverage_error_values)
    np.save("%s/auce_%s_empirical_coverage_negative_error.npy" % (save_dir, output), 
            neg_coverage_error_values)