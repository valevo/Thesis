# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles, corpora_from_pickles
from data.corpus import Sentences

from stats.stat_functions import compute_ranks, compute_freqs,\
                            pool_ranks, pool_freqs,\
                            reduce_pooled, merge_to_joint

from stats.mle import Mandelbrot                            
from stats.entropy import typicality

from jackknife.plotting import hexbin_plot, colour_palette, plot_preds


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette(colour_palette)




def get_greater_lims(lims1, lims2):
    if lims1 is None:
        return lims2
    (xlo1, xhi1), (xlo2, xhi2) = lims1[0], lims2[0]
    (ylo1, yhi1), (ylo2, yhi2)= lims1[1], lims2[1]
    return ((min(xlo1, xlo2), max(xhi1, xhi2)),
            (min(ylo1, ylo2), max(yhi1, yhi2)))
    
    
def mean_rank_freq_from_samples(sample_ls):
    rand_perm = np.random.permutation(sample_ls)
    half = len(sample_ls) // 2
    samples1, samples2 = rand_perm[:half], rand_perm[half:]

    ranks = [compute_ranks(sub) for sub in samples1]
    ranks_joined = pool_ranks(ranks)
    mean_ranks = reduce_pooled(ranks_joined)

    freqs = [compute_freqs(sub) for sub in samples2]
    freqs_joined = pool_freqs(freqs)
    mean_freqs = reduce_pooled(freqs_joined)
    return mean_ranks, mean_freqs


def within_filter_plots(sample_dict, show=True, mle_dict=None):
    plot_lims = None
    for i, (param, sample_ls) in enumerate(sample_dict.items()):
        mean_ranks, mean_freqs = mean_rank_freq_from_samples(sample_ls)
        joints = merge_to_joint(mean_ranks, mean_freqs)
        xs, ys = list(zip(*sorted(joints.values())))
        
        cur_plot_lims =\
        hexbin_plot(xs, ys, 
                    xlbl="$\log$ $r(w)$", ylbl="$\log$ $f(w)$", label=str(param), 
                    color=colour_palette[i], edgecolors=colour_palette[i],
                    linewidths=1.0, lims=None, min_y=1,
                    cbar=False)
        if mle_dict and param in mle_dict:
            mandelbrot = mle_dict[param]
            plot_preds(mandelbrot, np.asarray(xs), color=colour_palette[i])

        plot_lims = get_greater_lims(plot_lims, cur_plot_lims)
        print(plot_lims)
    
    
    plt.xlim(plot_lims[0])
    plt.ylim(plot_lims[1])
    plt.legend()
    if show:
        plt.show()
        
    return plot_lims
        
    

def across_filter_plots(tf_samples, srf_samples, f, h, uni_samples, show=False):
    tf_mean_ranks, tf_mean_freqs = mean_rank_freq_from_samples(tf_samples)
    srf_mean_ranks, srf_mean_freqs = mean_rank_freq_from_samples(srf_samples)
    uni_mean_ranks, uni_mean_freqs = mean_rank_freq_from_samples(uni_samples)
    
    tf_mean_rf = mean_rank_freq_from_samples(tf_samples)
    srf_mean_rf = mean_rank_freq_from_samples(srf_samples)
    uni_mean_rf = mean_rank_freq_from_samples(uni_samples)
        
    plot_lims = None
    
    joints = merge_to_joint(*uni_mean_rf)
    xs, ys = list(zip(*sorted(joints.values()))) 
        
    cur_plot_lims = hexbin_plot(xs, ys, xlbl="$\log$ $r(w)$", ylbl="$\log$ $f(w)$",
                                label="UNIF", 
                                color="black",
                                edgecolors="black",
                                cmap="gray",
                                linewidths=1.0, lims=None, min_y=1, cbar=True)
    plot_lims = get_greater_lims(plot_lims, cur_plot_lims)
    
    
    joints = merge_to_joint(*tf_mean_rf)
    xs, ys = list(zip(*sorted(joints.values()))) 
        
    cur_plot_lims = hexbin_plot(xs, ys, 
                                label="TF " + str(f), 
                                color=colour_palette[0],
                                edgecolors=colour_palette[0],
                                linewidths=1.0, lims=None, min_y=1, cbar=False)
    plot_lims = get_greater_lims(plot_lims, cur_plot_lims)
    
    
    joints = merge_to_joint(*srf_mean_rf)
    xs, ys = list(zip(*sorted(joints.values()))) 
        
    cur_plot_lims = hexbin_plot(xs, ys, 
                                label="SRF " + str(h), 
                                color=colour_palette[1],
                                edgecolors=colour_palette[1],
                                linewidths=1.0, lims=None, min_y=1, cbar=False)
    plot_lims = get_greater_lims(plot_lims, cur_plot_lims)
    
        
    plt.xlim(plot_lims[0])
    plt.ylim(plot_lims[1])
    plt.legend()
    if show:
        plt.show()



def get_reference_dist(wiki):
    n = int(10e6)
    m = 10

    wiki_ls = list(wiki)

    subsamples = [Sentences.subsample(wiki_ls, n) for _ in range(m)]
    mean_ranks, mean_freqs = mean_rank_freq_from_samples(subsamples)        
    joints = merge_to_joint(mean_ranks, mean_freqs)
    xs, ys = list(zip(*sorted(joints.values())))    
    mandelbrot = Mandelbrot(ys, xs)
    mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)    
    mandelbrot.register_fit(mandelbrot_fit)
    mandelbrot.print_result()
    return mandelbrot, mean_ranks


def samples_to_typicality(samples, ref_dist, rank_dict):
    freqs = [compute_freqs(s) for s in samples]
    joints = [merge_to_joint(rank_dict, f_dict) for f_dict in freqs]
    typs = [typicality(ref_dist, j) for j in joints] 
    return typs


def typicality_distributions(tf_dict, srf_dict, unis, ref_dist, rank_dict):
    tf_typ_dict = {param: samples_to_typicality(samples, ref_dist, rank_dict)
                    for param, samples in tf_dict.items()}
    
    srf_typ_dict = {param: samples_to_typicality(samples, 
                                                 ref_dist, rank_dict)
                    for param, samples in srf_dict.items()}
    
    uni_typs = samples_to_typicality(unis, ref_dist, rank_dict)
    
    
    hist_args = dict(alpha=1.0)
    for param, typs in tf_typ_dict.items():
        sns.distplot(typs, label="TF " + str(param), hist_kws=hist_args)
    
    for param, typs in srf_typ_dict.items():
        sns.distplot(typs, label="SRF " + str(param), hist_kws=hist_args)    
        
    sns.distplot(uni_typs, label="UNIF", 
                 axlabel="$a(C^k; P_{\hat{\\alpha}, \hat{\\beta}})$", hist_kws=hist_args)
    
    plt.legend()
#    plt.show()
    
    tf_mean_std_typ = {param: (np.mean(typs), np.var(typs)**.5)
                        for param, typs in tf_typ_dict.items()}
    srf_mean_std_typ = {param: (np.mean(typs), np.var(typs)**.5)
                        for param, typs in srf_typ_dict.items()}
    
    uni_mean_std_typ = np.mean(uni_typs), np.std(uni_typs)

    return tf_mean_std_typ, srf_mean_std_typ, uni_mean_std_typ


def do_mles(tf_samples, srf_samples, uni_samples):
    tf_mles = {}
    srf_mles = {}
    for param, sample_ls in tf_samples.items():
        print("\n TF", str(param))
        mean_ranks, mean_freqs = mean_rank_freq_from_samples(sample_ls)        
        joints = merge_to_joint(mean_ranks, mean_freqs)
        xs, ys = list(zip(*sorted(joints.values())))    
        mandelbrot = Mandelbrot(ys, xs)
        mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                        method="powell", full_output=True)    
        mandelbrot.register_fit(mandelbrot_fit)
#        result_str = mandelbrot.print_result(string=True)
        tf_mles[param] = mandelbrot
        
    for param, sample_ls in srf_samples.items():
        print("\n SRF", str(param))
        mean_ranks, mean_freqs = mean_rank_freq_from_samples(sample_ls)        
        joints = merge_to_joint(mean_ranks, mean_freqs)
        xs, ys = list(zip(*sorted(joints.values())))    
        mandelbrot = Mandelbrot(ys, xs)
        mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                        method="powell", full_output=True)    
        mandelbrot.register_fit(mandelbrot_fit)
#        mandelbrot.print_result()
        srf_mles[param] = mandelbrot
        
    uni_mean_ranks, uni_mean_freqs = mean_rank_freq_from_samples(uni_samples)
    uni_joints = merge_to_joint(uni_mean_ranks, uni_mean_freqs)
    uni_xs, uni_ys = list(zip(*sorted(uni_joints.values())))
    uni_mandelbrot = Mandelbrot(uni_ys, uni_xs)
    uni_mandelbrot_fit = uni_mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)    
    uni_mandelbrot.register_fit(uni_mandelbrot_fit)

    return tf_mles, srf_mles, uni_mandelbrot   
   

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", type=str)
    p.add_argument("--factors", nargs="*", type=int, default=[])
    p.add_argument("--hist_lens", nargs="*", type=int, default=[])
    
    args = p.parse_args()
    return args.lang, args.factors, args.hist_lens


def get_filters(filter_dir, k, names, param_name, param_ls):
    filters_dict = {}
    
    for param in param_ls:
        all_samples = corpora_from_pickles(filter_dir, names=names)
        cur_param_filters = [Sentences(c) for name_d, c in all_samples if 
                             name_d["k"] == k and name_d[param_name] == param]
        
        filters_dict[param] = cur_param_filters
        
    return filters_dict


if __name__ == "__main__":
    lang, factors, hist_lens = parse_args()
    print("ARGS: ", lang, factors, hist_lens, "\n")
    d =  "results/" + lang + "/"
    results_d = d + "evaluation/"

    k = 1000000
    
    srfs = get_filters(d + "SRF/", k, ["k", "h", "i"], "h", hist_lens)
    tfs = get_filters(d + "TF/", k, ["k", "f", "i"], "f", factors)

    highest_three_factors = factors[-3:]
    three_tfs = {k: tfs[k] for k in highest_three_factors}
    highest_three_hist_lens = hist_lens[-3:]
    three_srfs = {k: srfs[k] for k in highest_three_hist_lens}

    unis = [Sentences(c) for _, c in corpora_from_pickles(d + "UNI", names=["k", "i"])]
    
    uni_mean_ranks, uni_mean_freqs = mean_rank_freq_from_samples(unis)
    uni_joints = merge_to_joint(uni_mean_ranks, uni_mean_freqs)
    uni_xs, uni_ys = list(zip(*sorted(uni_joints.values())))

    print("filters loaded", flush=True)
    
    
    
    # MLEs
    tf_mles, srf_mles, uni_mandel = do_mles(tfs, srfs, unis)
    
    with open(results_d + "mle_mandelbrot.txt", "w") as handle:
        for param, mandel in tf_mles.items():
            handle.write("TF " + str(param))
            handle.write("\n")
            handle.write(mandel.print_result(string=True))
            handle.write("\n\n")
        for param, mandel in srf_mles.items():
            handle.write("SRF " + str(param))
            handle.write("\n")
            handle.write(mandel.print_result(string=True))
            handle.write("\n\n")

        handle.write("\nUNI\n")
        handle.write(uni_mandel.print_result(string=True))
        handle.write("\n\n")

    print("MLEs done")
    
    
    
    # within filter comparisons
    # TF
    uni_plot_lims = hexbin_plot(uni_xs, uni_ys, label="UNIF", linewidths=1.0,
                color="black", edgecolors="black", cmap="gray", alpha=0.5,
                cbar=True, min_y=1)    
    plot_lims = within_filter_plots(three_tfs,show=False)
    plot_lims = get_greater_lims(uni_plot_lims, plot_lims)
    plt.xlim(plot_lims[0])
    plt.ylim(plot_lims[1])
    plt.savefig(results_d + "TF_within_comp_rank_freq.png", dpi=300)
    plt.close()
    
    
    # SRF
    uni_plot_lims = hexbin_plot(uni_xs, uni_ys, label="UNIF", linewidths=1.0,
                color="black", edgecolors="black", cmap="gray", alpha=0.5,
                cbar=True, min_y=1)    
    plot_lims = within_filter_plots(three_srfs,show=False)
    plot_lims = get_greater_lims(uni_plot_lims, plot_lims)
    plt.xlim(plot_lims[0])
    plt.ylim(plot_lims[1])
    plt.savefig(results_d + "SRF_within_comp_rank_freq.png", dpi=300)
    plt.close()
    
    
    # within filter comparisons WITH MLEs
    # TF
    uni_plot_lims = hexbin_plot(uni_xs, uni_ys, label="UNIF", linewidths=1.0,
                color="black", edgecolors="black", cmap="gray", alpha=0.5,
                cbar=True, min_y=1)    
    plot_lims = within_filter_plots(three_tfs, show=False, mle_dict=tf_mles)
    plot_lims = get_greater_lims(uni_plot_lims, plot_lims)
    plt.xlim(plot_lims[0])
    plt.ylim(plot_lims[1])
    plt.savefig(results_d + "TF_within_comp_rank_freq_MLE.png", dpi=300)
    plt.close()
    
    # SRF
    uni_plot_lims = hexbin_plot(uni_xs, uni_ys, label="UNIF", linewidths=1.0,
                color="black", edgecolors="black", cmap="gray", alpha=0.5,
                cbar=True, min_y=1)    
    plot_lims = within_filter_plots(three_srfs, show=False, mle_dict=srf_mles)
    plot_lims = get_greater_lims(uni_plot_lims, plot_lims)
    plt.xlim(plot_lims[0])
    plt.ylim(plot_lims[1])
    plt.savefig(results_d + "SRF_within_comp_rank_freq_MLE.png", dpi=300)
    plt.close()
    

    
    print("compared within", flush=True)
    

    # across filter comparisons
    max_f, max_h = max(factors), max(hist_lens)
    across_filter_plots(tfs[max_f], srfs[max_h], max_f, max_h, unis)
    plt.savefig(results_d + "across_comp_rank_freq.png", dpi=300)
    plt.close()
    
    print("compared across", flush=True)    
    
    # typicality distributions
    wiki_iter = wiki_from_pickles("data/" + lang + "_pkl")
    ref_dist, big_mean_ranks = get_reference_dist(wiki_iter)
    tf_means, srf_means, (uni_mean, uni_std) = typicality_distributions(tfs, srfs, 
                                                              unis, ref_dist,
                                                              big_mean_ranks)
    plt.savefig(results_d + "typicality_distribution.png", dpi=300)
    plt.close()
    
    
    with open(results_d + "typicality_mean_stddev.txt", "w") as handle:
        for param, (mean_typ, std_typ) in tf_means.items():
            handle.write("\nTF " + str(param))
            handle.write("\t" + str(round(mean_typ, 3)) + "\t" + str(round(std_typ, 3)))
        for param, (mean_typ, std_typ) in srf_means.items():
            handle.write("\nSRF " + str(param))
            handle.write("\t" + str(round(mean_typ, 3)) + "\t" + str(round(std_typ, 3)))
        handle.write("\nUNI")
        handle.write("\t"+ str(round(uni_mean, 3)) + "\t" + str(round(uni_std, 3)))
        
        
    print("typicality distributions done", flush=True)