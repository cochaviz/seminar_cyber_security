---
title: Inversion Attacks on Federated Learning
author: Zohar Cochavi
bibliography: 
  - bibliography.bib
geometry: margin=2cm
documentclass: IEEEtran
classoption:
  - a4paper
abstract: This is the abstract...
---

<!-- **Abstract**

Federated Learning has brought improvements to the current state of machine
learning by providing better safety with regard to user data and privacy.
One of the ways in which this type of learning avoids hints about
releasing training data is by providing a layer of obscurity between the output
and the relation to the training data. Still, these models have been shown to be
able to be inverted. The impact of this on privacy and safety depends on the
extent to which this can be related to the individual training scheme. We
provide an overview of inversion attacks that can aid in the establishment of
risk assessments and (state-of-the-art something about why collecting is a good
idea) to determine whether these kinds of attacks pose a real threat in the
status-quo. -->

# Introduction

- [ ] why inversion attacks on federated learning are important
  - [ ] why inversion attacks in particular
  - [ ] why federated learning in particular
    - [ ] what does federated learning try to solve
    - [ ] why should we care about its safety

=> We know why we should be concerned with model inversion on federated learning

-----

Machine learning models have shown an incredible capacity to interpret data and
deduct from it, enabling numerous breakthroughs by improving diagnostic
capabilities in the medical field [@], natural language processing [@], etc.
All these applications, however, require privacy-sensitive data, which has been
shown to be inferrable from trained models in something called _Model Inversion_
[@fredriksonModelInversionAttacks2015]. Federated Learning attempts to address
this and other privacy-related security issues but is of course not infallible
to exploits [@abadSecurityPrivacyFederated2022]. Federated Learning is
relatively new to the space of machine learning (being introduced in 2015) [@],
and is already being used in large enterprises and essential infrastructure. For
this reason, cyber attacks such as Model Inversion could pose a significant,
unknown, threat to users (indirectly) using Federated Learning.

In this essay, I provide an overview of current Model Inversion threats and
defenses to Federated Learning. This overview should serve as an introduction to
Federated Learning and Model Inversion, and assist in the construction of risk
analyses in the context of Federated Learning. First, I will provide a general
overview of Model Inversion methods and techniques, and introduce Federated
Learning and some of the current threats. This is followed by an overview of
different, recent, Model Inversion attacks on Federated Learning and their
respective defenses. Finally, we will discuss how these developments have
influenced the threat landscape over the last year.

# Background

<!-- What is the goal of this section and what does it contain -->

In this section, the necessary background information will be introduced...

## Machine Learning

The goal of any machine learning algorithm is to predict the most likely outcome
given some, before unseen, set of features. All machine learning algorithms can
essentially be described by the following function [@abadSecurityPrivacyFederated2022]:

$$f(x): \mathbb{R}^d \rightarrow \mathbb{R}^{|C|}$$

Given several real-valued features, $d$, the algorithm returns the
probability of a certain class $c \in C$ occurring. If the number of classes,
$|C|$, is larger than 1, we often simply return the most likely class, also known
as classification. In the other case, $|C| = 1$, we either determine whether an
input is in that class (essentially returning a boolean value) or simply return
the value itself. The former is another form of classification, albeit a simpler
one, and the latter is regression.

When 'learning' the function, $f$, the algorithm uses a training set $X$ and
some sort of goal, often class labels, to determine how close its current
approximation of $f$, $\tilde{f}$, is to the goal. While, ideally, $\tilde{f}$
should be as general as possible would, and thus would eventually
equal $f$, this is never really the case [@]. In practice, the function will
remain an approximation that embeds some information from the training
set in its model to perform the approximation.

## Model Inversion

- [ ] What is model inversion
- [ ] How does it relate to different attacks

=> We can **explain how model inversion works** and **describe the
risk profile** in different (practical) contexts

-----

In _Model Inversion_, an attacker exploits this tendency to infer (possibly
sensitive) training data from a trained model. One notable example is that of
[@fredriksonModelInversionAttacks2015], who could, given the name of an
individual on which the model was trained, reconstruct faces by probing a facial
recognition model to a relatively successful degree ([@fig:inv_example]).

<div id="fig:inv_example">
![Reconstruction](images/images-002.png){#fig:inv_rec width=20%}
![Original](images/images-003.png){#fig:inv_original width=20%}

Face reconstruction from a trained model without access to training data.
</div>

The strength of any attack, however, is dependent on the information available
to the adversary. In the last example, the researchers considered a _black-box_
scenario. Meaning that they had no access to any of its (hyper-) parameters, but
could only probe it by submitting some input and observing the return values. In
the case of full access to a model and its (hyper-) parameters, a _white-box_
attack, one can imagine the potential of the attacker to be much greater.

In practice, performing a black-box model inversion attack often means
observing and 'cloning' a model [@] on which the actual attack is then
performed. This might seem convoluted, but often the search space is simply too
large to perform a brute-force attack without some sort of bootstrapping.

## Federated Learning

- [ ] what is federated learning
  - [ ] what does it solve
  - [ ] how does it work
- [ ] current threats to federated learning
  - [ ] how is federated learning currently used
  - [ ] have there been any successful attacks

=> We can **explain how Federated Learning works**, **describe the
problems it tries to solve**, and **describe the current threat landscape**
(this last one is slightly weaker)

-----

Federated Learning is a method of delegating, or democratizing the training
stage of a machine learning algorithm. Originally, it was created by Google to
avoid the problem of having users send privacy-sensitive data directly to them,
while reducing the amount of resources required to train the model which relies
on said data. Instead of directly learning a model on data, each user trains a
model on their own data and then the sends information, or _gradient_, of the
trained model to the central authority that will _aggregate_ these gradients
from multiple clients to produce the final model.

<!-- More technical details the type of data being sent -->

<!-- More technical detail on the aggregation process -->

From the perspective of cyber security, there are various moments at which one
can attack the 

# Model Inversion and Federated Learning

## Attacking

- [ ] Why is model inversion, in particular, a threat to Federated Learning
- [ ] State-of-the-art of model inversion in Federated Learning

=> **Describe the state-of-the-art in Federated Learning Model
inversion** and **describe the risk profile of each**.

-----

### Gradient Inversion

Federated learning (FL) allows the collaborative training
of AI models without needing to share raw data. This capability makes it
especially interesting for healthcare applications where patient and data
privacy is of utmost concern.  However, recent works on the inversion of deep
neural networks from model gradients raised concerns about the security of FL in
preventing the leakage of training data. In this work, we show that these
attacks presented in the literature are impractical in FL use-cases where the
clients’ training involves updating the Batch Normalization (BN) statistics and
provide a new baseline attack that works for such scenarios. Furthermore, we
present new ways to measure and visualize potential data leakage in FL.
[@hatamizadehGradientInversionAttacks2023]

## Defending

- [ ] How can some of the aforementioned attacks be prevented or their risks
reduced
- [ ] How effective are these defenses
- [ ] What is the trade-off

=> We can **describe some tactics to mitigate the impact of the
aforementioned attacks** and **use them in the context of a cost/benefit
analysis**

-----

### Image Augmentation

Federated Learning (FL) is a technology that facilitates a
sophisticated way to train distributed data. As the FL does not expose sensitive
data in the training process, it was considered privacy-safe deep learning.
However, a few recent studies proved that it is possible to expose the hidden
data by exploiting the shared models only. One common solution for the data
exposure is differential privacy that adds noise to hinder such an attack,
however, it inevitably involves a trade-off between privacy and utility. This
paper demonstrates the effectiveness of image augmentation as an alternative
defense strategy that has less impact on the trade-off. We conduct comprehensive
experiments on the CIFAR-10 and CIFAR-100 datasets with 14 augmentations and 9
magnitudes. As a result, the best combination of augmentation and magnitude for
each image class in the datasets was discovered. Also, our results show that a
well-fitted augmentation strategy can outperform differential privacy.

# Discussion

# Conclusion

- [ ] How 'risky' are the tactics described in this essay
- [ ] Do the defenses have any real impact on the risk of these attacks
- [ ] What would we need to research to improve any of these risks

=> We can **describe the overall risk of model inversion attacks on
federated learning** and **build on top of this paper to reduce their risks and
improve our understanding**

-----

# References
