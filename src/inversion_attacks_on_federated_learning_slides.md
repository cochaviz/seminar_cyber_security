---
title: Inference Attacks on Federated Learning - A Survey
author: Zohar Cochavi
theme: tud
date: June 27, 2023
bibliography: src/bibliography.bib
output: beamer_presentation
institute: Delft University of Technology, The Netherlands
---

# Introduction

## Relevance

- Federated Learning has seen large-scale use since its introduction
- Inference attacks threaten one of its core principles
- The field changes quickly, and updates are necessary

## Overview

1. Introduce Federated Learning and Adversarial Machine Learning
2. Discuss progress in the field over the last year (continuing work by @abadSecurityPrivacyFederated2022)
3. Conclude what this means for Federated Learning

# Background

We will cover:

1. Federated Learning
2. Inference Attacks

# Federated Learning {.allowframebreaks}

_Federated Learning_ (FL) is a machine learning scheme that distributes the
responsibility of training a model over multiple clients and aggregates their
results into a single model [@mcmahanFederatedLearningCollaborative2017].

- $\uparrow$ Better privacy guarantees
- $\uparrow$ Distribution of resources
- $\downarrow$ More resources in total

![Typical Federated Learning network topology. The client, $c_i$, sends the
gradient, $\nabla Q(w_i)$, and/or weights, $w_i$, of a particular
iteration $i$.  The central server then returns the updated model parameters
$w_{i+1}$.](../images/client-server-fl.svg){width=80%}

# Inference Attacks {.allowframebreaks}

We categorize inference attacks according to the following properties [@abadSecurityPrivacyFederated2022].

## Adversarial goal

- _Model Inversion_: find data points by the label.
- _Membership Inference_: determine the presence of a data point in the local
  training data.
- _Property Inference_: determine presence property $p$ of the data or model.

<!-- TODO: Include example? -->

## Interference with learning

- _Passive_: does not interfere with the learning process.
- _Active_: interferes with the learning process to gain more information.

_Passive_ attacks are more stealthy, but _active_ attacks are stronger.

## Position of the adversary

- _Local_: adversary is a client.
- _Global_: adversary is the central server.

Often, the information required is available to both and an attack considers a
_local/global_ scenario. Meaning it could be both.

# Inference Attacks in Federated Learning

We will discuss three of the most interesting ones.

1. Do Gradient Inversion Attacks Make Federated Learning Unsafe?
2. Active Membership Inference Attack under Local Differential Privacy in
Federated Learning.
3. Subject Membership Inference Attacks in Federated Learning

# Do Gradient Inversion Attacks Make Federated Learning Unsafe?

- @hatamizadehGradientInversionAttacks2023 explore image reconstruction
using gradient inversion while relaxing the assumption made in prior work
regarding Batch Normalization (BN).

- Previous studies assumed static BN statistics, but the authors successfully
reconstructed images without relying on this assumption.

- Inversion attacks can be practical for accurate reconstructions but still
require priors (approximations of the image) for higher accuracy.

$\Rightarrow$ Attack that more closely resembles a real-world scenario.

# Active Membership Inference Attack under Local Differential Privacy in Federated Learning.

- @nguyenActiveMembershipInference2023 introduces an _active_ membership
inference attack, allowing them to infer membership of a specific data point in
the presence of differential privacy.

- Differential privacy is a technique that obscures an individual's relation to
a data point while preserving the patterns used for training machine learning
models [@dworkAlgorithmicFoundationsDifferential2013].

- The attack performance starts to degrade only when the level of data obscuring
interferes with the model's performance, indicating the need for more robust
privacy methods to counter such attacks.

$\Rightarrow$ Raises questions about the efficacy of Differential Privacy.

# Subject Membership Inference Attacks in Federated Learning

- In a black-box setting, the paper by [@suriSubjectMembershipInference2022]
proposes a method called "Subject Inference" for inferring the presence of
individuals, or "subjects," in a dataset.

- Previous work in this area is criticized for being disconnected from
real-world scenarios as it includes information adversaries would not normally
have access to and assumes the adversary is looking for data points rather than
individuals.

- The authors demonstrate the effectiveness of Subject Inference in various
real-world datasets, emphasizing its realistic nature and highlighting it as a
significant threat to user privacy.

$\Rightarrow$ An attack crafted to reflect a real-life scenario for a
_cross-silo_ FL configuration.

# Defenses

Various novel defenses are proposed,

- The use of image augmentation to enhance privacy [@shinEmpiricalAnalysisImage2023]
- Using a built-in adversary [@liResSFLResistanceTransfer2022]

as well as suggestions to counterattack proposed attacks,

- Increase batch size to mask local contributions
  [@gengImprovedGradientInversion2023; @hatamizadehGradientInversionAttacks2023]
- Use alternative aggregation methods such as FedAvg and FedBN
  [@gengImprovedGradientInversion2023; @hatamizadehGradientInversionAttacks2023]

Another promising option that has not been investigated in this work is
_Homomorphic Encryption_ [@leePrivacyPreservingMachineLearning2022].

# Future Work

1. **Utilize existing preprocessing methods to enhance privacy preservation**, as
demonstrated by studies such as @shinEmpiricalAnalysisImage2023. Use
generalization to the advantage of privacy.

2. **New attack methods should prioritize relaxing assumptions** to provide a
more realistic assessment of privacy-preserving features in Federated Learning
(FL).

3. **Developing secure Homomorphic Encryption (HE) techniques would
significantly mitigate many of the attacks discussed**. Encrypting data before
training models would render inference attacks harmless.

# Conclusion

- Threats to current Federated Learning because of more realistic scenarios
- Privacy Enhancing technologies can be circumvented
- More research is necessary to assess whether FL is adequately privacy-preserving

# References {.allowframebreaks}
