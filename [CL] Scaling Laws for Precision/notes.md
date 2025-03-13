# Scaling Laws for Precision
    URL: https://arxiv.org/abs/2411.04330

    ABS: Low precision training and inference affect both the quality and cost of language models, but current scaling laws do not account for this. In this work, we devise “precision-aware” scaling laws for both training and inference. We propose that training in lower precision reduces the model’s effective parameter count, allowing us to predict the additional loss incurred from training in low precision and post-train quantization. For inference, we find that the degradation introduced by post-training quantization increases as models are trained on more data, eventually making additional pretraining data actively harmful. For training, our scaling laws allow us to predict the loss of a model with different parts in different precisions, and suggest that training larger models in lower precision may be compute optimal. We unify the scaling laws for post and pretraining quantization to arrive at a single functional form that predicts degradation from training and inference in varied precisions. We fit on over 465 pretraining runs and validate our predictions on model sizes up to 1.7B parameters trained on up to 26B tokens.

## 1. Introduction
1. "training larger models in lower precision can be compute-optimal"

This is mentioned in many papers.

## 2. Background, Related Work, and Setup
1. "Quantization: What. Only weights. “Quantization-aware training” Quantizing only weights during training does not offer any compute savings because matrix multiplications are still done in high precision. However, this is commonly done to allow weights to adapt to low precision so they can be served at very low precision at inference-time, thereby alleviating memory bottlenecks (Ma et al., 2024; Wang et al., 2023). We will refer to this as “quantization-aware-training.”"

Good to know. High precision models dealing with low precision problem may cause some issue.

## 3. Scaling Laws for Post-Train Quantization
1. "Loss degradation from PTQ increases with data."

As mentioned above.

2. "The intuition for this poor data scaling might be that as models train on more data, they compress more information into their weights, so that perturbations to weights in the form of quantization are more harmful to loss, all else equal."

Like 2.1 exactly

## 4. Scaling Laws for Quantized Training
1. "The effects of quantizing the weights, activations ,and KV cache during training are well modeled as independent and multiplicative."

2. "... which suggests as precision of training decreases at fixed compute, we should increase parameters and decrease data."

3. " If You Must Train in Low Precision, Increase Parameters Before Data."

4. " ButCompute-Optimal Pretraining Precision Can Increase in Compute if Model Size N is Constrained."

## 5. an Unifiled Scaling Law for Precision
1. "We observe that degradation very quickly increases to its exponentially large value from Section 3 if there is any gap between training and inference-time precision."

So fast.

