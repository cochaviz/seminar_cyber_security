---
title:  Inference Attacks on Federated Learning - A Survey
author: Hoar Cohabit
bibliography: 
  - bibliography.bib
documentclass: elsarticle
classoption:
  - 5p
abstract: This is the abstract...
---

  
> Federated Learning has brought improvements to the current state of machine
> learning by providing better safety with regard to user data and privacy. Gradient
> inversion and other inference attacks threaten on of the fundamental
> principles on which this method was founded.
>
> One of the ways in which this type of learning avoids hints about releasing
> training data is by providing a layer of obscurity between the output and
> the relation to the training data. Still, these models have been shown to be
> able to be inverted. The impact of this on privacy and safety depends on the
> extent to which this can be related to the individual training scheme. We
> provide an overview of inversion attacks that can aid in the establishment
> of risk assessments and (state-of-the-art something about why collecting is
> a good idea) to determine whether these kinds of attacks pose a real threat
> in the status-quo.

# Introduction

> - [ ] why inversion attacks on federated learning are important
> - [ ] why inversion attacks in particular
> - [ ] why federated learning in particular
>   - [ ] what does federated learning try to solve
>   - [ ] why should we care about its safety
>
> $\Rightarrow$ We know why we should be concerned with model inversion on
> federated learning

Machine learning models have shown an incredible capacity to interpret data and
deduct from it, enabling numerous breakthroughs by improving diagnostic
capabilities in the medical field [@], natural language processing [@], etc.
All these applications, however, require privacy-sensitive data, which has been
shown to be inferrable from trained models
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

> - [ ] Introduce Federated Learning from basic principles
> - [ ] Introduce machine learning exploits, focussing on inference attacks
> - [ ] Cover the state of FL inference up until last year
>
> $\Rightarrow$ We have a good understanding of FL and inference attacks

In this section, the necessary background information will be introduced.
The background is considered whatever already existed until last year (March
2022). We will provide a concise overview of machine learning principles, to
then discuss the workings of Federated Learning (FL). Having covered the
necessary machine learning knowledge, the discussion will move to how one would
attack such systems. Finally, we focus on previous inference attacks as
summarized and discussed by [@abadSecurityPrivacyFederated2022].

## Machine Learning

The goal of any machine learning algorithm is to predict some label or
value given familiar but unseen data. For the purposes of this discussion,
the machine-learning process can be separated into 3 stages:

1. Training
2. Testing/Evaluation
3. Deployment

During training, the machine learning model, $f$ is given a set of tuples $\{(x_i,
y_i)\}$. The learning algorithm then adjusts the model parameters, $\theta$,
such that it, $f_\theta$, maps the input features $x$ to the target value(s)
$y$. Depending on the learning task, $y$ could be a continuous value
(regression), a binary value (binary classification), or a set of discrete
values (multi-class classification) [@abadSecurityPrivacyFederated2022,
@chakrabortyAdversarialAttacksDefences2018].

<!-- TODO: Add a footnote that this is correct but not complete (unsupervised
learning etc.)? -->

The testing phase assesses the performance of the model. We take a similar, but
unseen set of tuples $\{(x, y)\}$ and test whether the model, $f_\theta(x)$
returns the correct value(s). It's important to only test on _unseen_ data since
the aim is to assess the _generalizing_ ability of the model. One can imagine
the performance to be higher if the same data that was used to train, was used
to test.

After reaching convergence, when the model does not improve further, it is
deployed to a public, or private, interface.

## Federated Learning

> - [ ] what is federated learning
>   - [ ] what does it solve
>   - [ ] how does it work
>
> $\Rightarrow$ We can **explain how Federated Learning works**, **describe the
> problems it tries to solve**, and **describe the current threat landscape**
> (this last one is slightly weaker)

Federated Learning (FL) is a method of delegating, or democratizing, the
training stage of a machine learning algorithm. Its benefits are threefold
[@konecnyFederatedLearningStrategies2017]:

1. It avoids sharing _raw_ personal data with a third party.
2. Processing resources are delegated.
3. Data that is fragmented over multiple locations can still be used for
   training.

The process, generally, works as follows. A set of clients, $C={c_0, \ldots,
c_n}$, with each their own dataset, train a machine learning model on that
dataset. Information about this trained model is then sent to a central server
that _aggregates_ the information from all clients into a single model. The
newly trained model is then sent back to the clients for another iteration
[@abadSecurityPrivacyFederated2022, @konecnyFederatedLearningStrategies2017].

Various aggregation algorithms exist. The most popular of which are _Federated
Stochastic Gradient Descent_ (FedSGD) and _Federated Averaging_ (FedAvg). These
both use the _gradient_ (the function that describes how to optimize the model)
or the aforementioned parameters, $\theta$, of the client models to optimize the
central model [@abadSecurityPrivacyFederated2022,
@guCSMIAMembershipInference2022]. While technical details of these algorithms
are not crucial to this discussion, it is important to note that this gradient
contains information about the distribution of the client's dataset.

<!-- TODO: Add a footnote that the aggregation methods also influence the local
training methods -->

Lastly, there are two types of FL: Horizontal Federated Learning (HFL), or
cross-device Federated Learning; and Vertical Federated Learning (VFL), or
cross-silo Federated Learning [@suriSubjectMembershipInference2022,
@abadSecurityPrivacyFederated2022]. In the first, devices all collect data on
the same features, but their sample space is not equal (the distributions might
not align, and the size can be different). In the latter, we can imagine a set
of hospitals, companies (or _data silos_) that need to train a model on _all_
the available user data. The data, however, cannot be directly shared and the
data they collect on their users is also different. Each user is present in each
database (they share the same sample space), but the features on each
measurement might be different. HFL is much more prevalent than VFL [@].

<!-- TODO: Mention that there are other network architectures for FL -->

## Attacking Machine Learning

The machine learning approaches discussed so far (normal/_centralized_
learning and Federated Learning) contain several points at which an
attacker could intervene to exploit various characteristics of the system.
Before discussing inference attacks that take place in later stages of the
machine learning pipeline, let us discuss briefly discuss other potential
threats to machine learning models.

The phases discussed in the first section (training, testing, and deployment)
correspond directly to different attacks which can be categorized as follows
[@chakrabortyAdversarialAttacksDefences2018].

1. _Poisoning Attack_: This type of attack, known as contamination of the
training data, takes place during the training time of the machine learning
model. An adversary tries to poison the training data by injecting carefully
designed samples to compromise the whole learning process eventually.
2. _Evasion Attack_: This is the most common type of attack in the adversarial
setting. The adversary tries to evade the system by adjusting malicious samples
during the testing phase. This setting does not assume any influence over the
training data.
3. _Inference/Exploratory Attack_: These attacks do not influence any dataset. Given
black-box access to the model, they try to gain as much knowledge as possible
about the learning algorithm of the underlying system and pattern in training
data.

While the first two also pose potential threats to the FL scheme and are very
popular in centralized machine learning, they are considerably harder to perform
on Federated Learning as the data is distributed
[@tolpeginDataPoisoningAttacks2020]. Databases of multiple clients have to be
compromised to create an exploit that is comparable to that of a centralized
machine-learning approach.

Inference attacks, however, threaten the privacy guarantees FL attempts to give.
Inference attacks specifically try to _infer_ information about the dataset the
model was trained on or the model itself. Thereby threatening the
confidentiality of the database, and thus the privacy, of the victims involved
[@abadSecurityPrivacyFederated2022]. Since they actively infer information about
a deployed system, the amount of information on the system determines how
powerful such an attack could be. For this reason, they are also further
specified as _white-box_ or _black-box_, and sometimes _grey-box_ inference
attacks [@nasrComprehensivePrivacyAnalysis2019].

<!-- TODO: This last sentence is iffy -->

## Inference Attacks

> - [ ] Give an overview of inference attack classifications
> - [ ] Provide an example of an inference attack
>
> $\Rightarrow$ We should **be able to classify different attacks, understand
> how they work**, and see **how they can compromise the privacy of an
> individual using the system**

Inference attacks can be applied to both centralized machine learning models and
Federated Learning schemes. Many of the principles we will cover apply to both
centralized and Federated Learning, but the focus will be on applications on FL.
Specifically, we will provide an overview of attack classifications as given by
[@abadSecurityPrivacyFederated2022].

Firstly, depending on the target information the attacker attempts to infer, the
attack is classified as follows:

- _Model Inversion_: In model inversion, the attacker attempts to invert the
machine learning model. Thereby finding the data point corresponding to a
certain label. [@fredriksonModelInversionAttacks2015] were able to invert a
facial recognition model, allowing them to recover the image of an individual
known to be in the training dataset.

- _Membership Inference_: In this attack, the goal is to determine whether a
data point $(x, y)$ was part of the training set. In FL it is also possible to
determine whether a data point was part of the training set of a particular
client.

- _Property Inference_: This attack leverages model snapshot updates concerning a
dataset without the objective property and another one containing it. Afterward,
a trained binary classifier can relate the property with an unseen update, e.g.,
an update got by eavesdropping, to conclude whether the update owns the
objective property. For a data piece $x$ containing the property $p: x^{(p)}$
belonging to the dataset $D_n$ being $n$ the number of clients, the attacker infers
if the property belongs to the dataset $x^{(p)} \in D_n$.

<!-- TODO: Should this Property Inference be included? -->

Secondly, when considering a malicious central server, the attack can be
classified according to the manner in which the attacker interferes with the
training procedure [@abadSecurityPrivacyFederated2022]:

- _Passive_: Also known as an _honest-but-curious_ scenario. The attacker can
only read or eavesdrop on communications (i.e. the weights and gradient
transmitted between the clients and the server), and the local model and
dataset in case of an honest-but-curious client.

- _Active_: The attack is more effective than the former but less stealthy.
It essentially changes the learning goal from minimizing loss to maximizing
inferred information from the victim.


# Inference Attacks in Federated Learning

The risk of any attack should be 

## Attacking

> - [ ] State-of-the-art of model inversion in Federated Learning
>
> $\Rightarrow$ **Describe the state-of-the-art in Federated Learning Model inversion**
> and **describe the risk profile of each**.

### Gradient Inversion

- Do Gradient Inversion Attacks Make Federated Learning Unsafe?[@hatamizadehGradientInversionAttacks2023]

Was able to infer images over multiple batches by inverting the gradient in an
"honest-but-curious" scenario. Previous work has made simplifying assumptions of
the training procedure, which the authors have been able to work around. One of
these assumptions was the lack of dynamism in the Batch Normalization procedure.
Essentially, previous work assumed the mean and variance of each batch to be
equal. This assumption is, however, not realistic. The authors have found a
method in which this assumption can be relaxed while finding similar results.
They considered both a **local** and **global** adversary.

- Improved Gradient Inversion Attacks and Defenses in Federated Learning
[@gengImprovedGradientInversion2023] 

Created a strategy for inverting both FedAVG-based and FedSGD-based networks in
an "honest-but-curious" scenario. (They say their results are the first
successful model inversion attack on FedAVG, but I have not verified this).


### Membership Inference

- CS-MIA: Membership inference attack based on prediction confidence series in
federated learning[@guCSMIAMembershipInference2022]

> In a local/global "honest-but-curious" setting, the authors determine whether
> data points are members of certain datasets by following the trend in their
> classification confidence. They train a supervised model on data points that are
> known to be members of a dataset (or not) based on the aforementioned
> assumption. This model is then used to determine the probability of unseen data
> to be in the target model. They show incredibly high accuracy and F1-scores for
> all datasets except MNIST, event thought it still scores the best out of all
> included approaches.

- Subject Membership Inference Attacks in Federated Learning [@suriSubjectMembershipInference2022]

> In a black-box setting, the authors propose a method for _subject-inference_ in
> a cross-silo, or vertical, FL setup. They argue previous work being disconnected
> from reality as they (i) include information adversaries would not normally have
> access to and (ii) redefine the goal of membership inference from a record-based
> one to a subject-based one. Instead of determining whether one particular
> data-point was part of the training set, we attempt to infer whether an
> individual (or rather their distribution) is present in the dataset given some
> preexisting information on them. They show these attacks to be a realistic
> threat for subject-level privacy in various different learning modes. 

- Active Membership Inference Attack under Local Differential Privacy in
Federated Learning [@nguyenActiveMembershipInference2023]

> Different from other works, the authors in this paper considered a completely
> malicious, or _active_ membership inference attack. In this attack, the central
> authority returns are carefully crafted response to the gradients submitted by
> the clients. This response is not intended to reduce the loss of any of the
> clients' models, but constructed purely to infer whether some data-point is a
> member of the training set.
 
## Defending

> - [ ] How can some of the aforementioned attacks be prevented or their risks
> reduced
>
> $\Rightarrow$ We can **describe some tactics to mitigate the impact of the
> aforementioned attacks** and **use them in the context of a cost/benefit
> analysis**

- CRFL: A novel federated learning scheme of client reputation assessment via
local model inversion [@zhengCRFLNovelFederated2022]

- An empirical analysis of image augmentation against model inversion attack in
federated learning [@shinEmpiricalAnalysisImage2023]

- ResSFL: A Resistance Transfer Framework for Defending Model Inversion Attack
in Split Federated Learning [@liResSFLResistanceTransfer2022]

# Discussion

> - [ ] How effective are these defenses
> - [ ] What is the trade-off

# Conclusion

> - [ ] How 'risky' are the tactics described in this essay
> - [ ] Do the defenses have any real impact on the risk of these attacks
> - [ ] What would we need to research to improve any of these risks
>
> $\Rightarrow$ We can **describe the overall risk of model inversion attacks on
> federated learning** and **build on top of this paper to reduce their risks
> and improve our understanding**

# References
