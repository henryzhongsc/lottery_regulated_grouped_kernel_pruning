# TMI-GKP: Revisit Kernel Pruning with Lottery Regulated Grouped Convolutions

> This is the official codebase for [our paper](https://openreview.net/forum?id=LdEhiMG9WLO) accepted at ICLR 2022. Should you need to cite our paper, please use the following BibTeX:

```
@inproceedings{
zhong2022revisit,
title={Revisit Kernel Pruning with Lottery Regulated Grouped Convolutions},
author={Shaochen Zhong and Guanqun Zhang and Ningjia Huang and Shuai Xu},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=LdEhiMG9WLO}
}
```

---

## Getting started (with a lightweight experiment on Colab)

There is no better way to familiarize a codebase than running an example trial. Thus, we provide a demo Google Colab [notebook](https://github.com/choH/lottery_regulated_grouped_kernel_pruning/blob/main/TMI_GKP_demo.ipynb) with annotation, where you are able to run the ResNet-20 on CIFAR-10 experiment with one-click as all necessary files are supplemented in this GitHub repo.

This Colab demo file will walk you through every stage of a typical TMI-GKP task: model initialization, baseline training, snapshot registration, pruning, fine-tuning, supplied with `setting.json` files for all tasks mentioned here.


For a `--task finetune` (where TMI-GKP plays), the basic logic is the `main.py` file will take a baseline model, the baseline model's training snapshot folder for TMI evaluation, and a pruning setting file. You will also need to specify an output folder, where upon completion of this `--task finetune` experiment, you should find:

* `baseline`: Copy of the baseline model
* `pruned`: The pruned model (with clustering information to extract a `cluster.json` as described in Section A.3.2 of the paper).
* `grouped`: The grouped model. This is basically `pruned` but in grouped convolution architecture (with sparsity removed). All non-inference-related portions are also removed (e.g., the clustering info).
* `finetuned`: The finetuned model trained upon `grouped`.
* `setting.json`: Copy of the assigned setting file, but with more information stored (e.g., configuration assigned via `argparse`, log of acc/loss, etc.). Should you want to rerun this experiment, you should pass the original setting file (in this case, `settings/resnet20_cifar10_prune_and_finetune.json`) but not this one.
* `experiment.log`: Experiment printouts registered by the logger. This will also print to the terminal for easier monitoring.
* `/checkpoints/` folder: Saved checkpoints during the finetuning process.

---

Please do not hesitate to open an issue for questions or discussions. Being my undergraduate work, I'd be first to admit — and there's no denying with the complete lack of docs and comments — this is not the cleanest codebase you'd work with, but I keep a close monitor of this repo, yet I put in serious efforts to help people as showcased in the issues we got.   

**We the authors believe *Grouped Kernel Pruning* is a promising direction for future structured pruning methods, as it makes sense to pursue a higher degree of pruning freedom yet retain those densely structured properties** (empirical results also suggest a one-shot adoption of GKP is often better than most iterative filter pruning methods with much higher retraining/finetuning budget). We'd hope more people from the pruning community may explore it in lieu of the already crowded filter/channel pruning realms.

---
---

## Demo experiments, and to reproduce experiments in Table 2.

As mentioned in Appendix 3.2 of our paper, we have replicated four CIFAR-10 experiments within the abovementioned Colab notebook; please refer to the [`demo_experiments`](https://github.com/choH/lottery_regulated_grouped_kernel_pruning/tree/main/demo_experiments) folder should you need to checkout or replicate.

To reproduce experiments in Table 2, you should first obtain the respective baseline model with necessary training snapshots. Then you will need to make a `setting.json` file that follows the structure of [`settings/resnet20_cifar10_reported.json`](https://github.com/choH/lottery_regulated_grouped_kernel_pruning/blob/main/settings/resnet20_cifar10_reported.json) (you should find the necessary information at Appendix 3.1 and Appendix 3.2). Last, you will need to mimic the `argparse` command of above ResNet-20 + CIFAR-10 experiment to kick-of the intended experiment. Do make sure that the assigned `output_folder_dir` is empty if you do not intend to overwrite.

---
---

### Discussion regarding inference speedup

Please checkout Appendix 3.3 of our paper before reading the following discussion.

2/5 reviewers ([`hEhG`](https://openreview.net/forum?id=LdEhiMG9WLO&noteId=GPAYsg6ryPS) and [`4Wb3`](https://openreview.net/forum?id=LdEhiMG9WLO&noteId=5Hj0pWsf_w_)) asked for inference speed up provided by our method. This is the part we have to unfortunately say no to a reasonable request. We did not, and will continue not to, provide a speedup analysis because of the following two reasons.

Speedup analysis is sensitive to the hardware and implementation of the network, so a proper speedup comparison would require us to replicate the code of other methods, which we simply don't have such manpower to do. In fact, among all compared methods at Table 2, the only one who replicates others' code (Wang et al. 2021) also opts to not include a speedup comparison. Out of 12 methods we compared against, only 5 of them have done a speed analysis and all of them are pre-2019.

Despite the limitation of speed up comparison, we'd agree that an "inner" speedup analysis between our pruned network and the original network is a very reasonable request. **Unfortunately, we can't deliver a reasonable result because `PyTorch` is slowing us down as the optimization on grouped convolution in torch is much slower than standard convolution.** This is likely because their current implementation of grouped convolution is simply performing the standard convolution channel-by-channel [2]. Please see these two GitHub issues ([issue_1](https://github.com/pytorch/pytorch/issues/10229), [issue_2](https://github.com/pytorch/pytorch/issues/18631)) and paper [1] and [2]. To provide an empirical prove, running [this test code](https://github.com/pytorch/pytorch/issues/18631#issuecomment-478155467) with `groups = 8`, the average speed of the grouped convolution is 20.29 times slower than the standard convolution despite the former one has less params and flops. See below output for details:

```
1000 loops, best of 5: 780 µs per loop
100 loops, best of 5: 16.4 ms per loop
```

The two potential solutions to this problem are a) implement our own cuda optimization, which is frankly beyond our ability and scope of the type of paper we submitted; or b) convert the torch model to a inference-focused framework `ONNX` and hopefully ONNX has a better support of grouped convolution. But unfortunately it doesn't seem to be the case according to this [issue](https://github.com/microsoft/onnxruntime/issues/9192).

**The good news is, both torch and ONNX are working on better grouped convolution optimization** as stated in the responses of their GitHub issues. As scholars already successfully accelerated grouped convolution across different machine learning platform [1, 2], it is reasonable to expect ML platforms to achieve the same goal, with similar or even better efficiency.

Practically, since a grouped convolution with `groups = n` can be implemented as `n` standard convolution and deploy on `n` edge devices in a parallel fashion [3], **our algorithm is still applicable in a production setting and will enjoy the benefits we mentioned in our manuscript.**

We hope this response may help the reviewers better aware the current status of speedup analysis on grouped convolutions, and understand why we are not able to provide such experiment.

[1] P. Gibson et al. Optimizing Grouped Convolutions on Edge Devices. ASAP 2020
[2] Z. Qin et al. Diagonalwise Refactorization: An Efficient Training Method for Depthwise Convolutions. IJCNN 2018
[3] Z. Su et al. Dynamic Group Convolution for Accelerating Convolutional Neural Networks. ECCV 2020
