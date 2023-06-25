---
title:  Inference Attacks on Federated Learning - A Survey
author: Zohar Cochavi

bibliography: 
  - bibliography.bib
documentclass: IEEEtran
classoption:
  - compsoc

abstract: Federated Learning has brought improvements to the current state of machine learning by providing better safety with regard to user data and privacy. Gradient inversion and other inference attacks threaten on of the fundamental principles on which this method was founded.  One of the ways in which this type of learning avoids hints about releasing training data is by providing a layer of obscurity between the output and the relation to the training data. Still, these models have been shown to be able to be inverted.  The impact of this on privacy and safety depends on the extent to which this can be related to the individual training scheme. We provide an overview of inversion attacks that can aid in the establishment of risk assessments and (state-of-the-art something about why collecting is a good idea) to determine whether these kinds of attacks pose a real threat in the status-quo.
---


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

In this section, the necessary background information will be introduced. The
background is considered whatever already existed until last year (March 2022).
We will provide a concise overview of machine learning principles, to then
discuss the workings of Federated Learning (FL). Having covered the necessary
machine learning knowledge, the discussion will move to how one would attack
such systems. Finally, we focus on previous inference attacks as summarized and
discussed by [@abadSecurityPrivacyFederated2022].

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
> $\Rightarrow$ We can **explain how Federated Learning works** and **describe the
> problems it tries to solve**. 

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
[@abadSecurityPrivacyFederated2022,@konecnyFederatedLearningStrategies2017].

![Typical Federated Learning network topology. The client, $c_i$, sends the
gradient, $\nabla Q(\theta_i)$, and/or weights, $\theta_i$, of a particular
iteration $i$.  The central server then sends the updated model parameters
$\theta_{i+1}$ initiating the next iteration.
](images/client-server-fl.png)

Various aggregation algorithms exist. The most popular of which are _Federated
Stochastic Gradient Descent_ (FedSGD) and _Federated Averaging_ (FedAvg). These
both use the _gradient_ (the function that describes how to optimize the model)
or the aforementioned parameters, $\theta$, of the client models to optimize the
central model
[@abadSecurityPrivacyFederated2022,@guCSMIAMembershipInference2022]. While
technical details of these algorithms are not crucial to this discussion, it is
important to note that this gradient contains information about the distribution
of the client's dataset.

<!-- TODO: Add a footnote that the aggregation methods also influence the local
training methods -->

Lastly, there are two types of FL: Horizontal Federated Learning (HFL), or
cross-device Federated Learning; and Vertical Federated Learning (VFL), or
cross-silo Federated Learning
[@suriSubjectMembershipInference2022,@abadSecurityPrivacyFederated2022]. In the
first, devices all collect data on the same features, but their sample space is
not equal (the distributions might not align, and the size can be different). In
the latter, we can imagine a set of hospitals, companies (or _data silos_) that
need to train a model on _all_ the available user data. The data, however,
cannot be directly shared and the data they collect on their users is also
different. Each user is present in each database (they share the same sample
space), but the features on each measurement might be different. HFL is much
more prevalent than VFL [@].

<!-- TODO: Improve explanation and include a reference to subject inference -->

<!-- TODO: Mention that there are other network architectures for FL -->

## Attacking Machine Learning

The machine learning approaches discussed so far (normal/_centralized_
learning and Federated Learning) contain several points at which an
attacker could intervene to exploit various characteristics of the system.
Before discussing inference attacks that take place in later stages of the
machine learning pipeline, let us briefly discuss other potential
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
@abadSecurityPrivacyFederated2022.

Firstly, depending on the target information the attacker attempts to infer, the
attack is classified as follows:

- _Model Inversion_: In model inversion, the attacker attempts to invert the
machine learning model. Thereby finding the data point corresponding to a
certain label. @fredriksonModelInversionAttacks2015 were able to invert a
facial recognition model, allowing them to recover the image of an individual
known to be in the training dataset.

- _Membership Inference_: In this attack, the goal is to determine whether a
data point $(x, y)$ was part of the training set. In FL it is also possible to
determine whether a data point was part of the training set of a particular
client.

- _Property Inference_: This attack leverages model snapshot updates concerning
a dataset without the objective property and another one containing it.
Afterward, a trained binary classifier can relate the property with an unseen
update, e.g., an update got by eavesdropping, to conclude whether the update
owns the objective property. For a data piece $x$ containing the property $p:
x^{(p)}$ belonging to the dataset $D_n$ being $n$ the number of clients, the
attacker infers if the property belongs to the dataset $x^{(p)} \in D_n$.

<!-- TODO: Should this Property Inference be included? -->

Secondly, when considering a malicious central server, the attack can be
classified according to the manner in which the attacker interferes with the
training procedure [@abadSecurityPrivacyFederated2022]:

- _Passive_: Also known as an _honest-but-curious_ scenario. The attacker can
only read or eavesdrop on communications (i.e. the weights and gradient
transmitted between the clients and the server), and the local model and dataset
in case of an honest-but-curious client.

- _Active_: The attack is more effective than the former but less stealthy.  It
essentially changes the learning goal from minimizing loss to maximizing
inferred information from the victim.

Lastly, attacks can be categorized based on the position of the attacker in the
network:

- _Local_: The attacker is a client, i.e. they can only access _their_ database,
parameters, and the global parameters they receive from the server.

- _Global_: The attacker is the central server. They do not have access to any
databases but can access the gradients/parameters sent by all clients and the
global model.

# Inference Attacks in Federated Learning

> - [ ] Describe state-of-the-art of model inversion in Federated Learning
> - [ ] Quickly determine how they relate to other attacks and interpret their
>   significance.
>
> $\Rightarrow$ Be able to **describe the state-of-the-art of inversion attacks
> on Federated Learning** and **describe the risk profile of each**.

This section will discuss the state-of-the-art of inference attacks on Federated
Learning. Specifically, we will discuss progress in the field as of March 2022.
The research presented here was found primarily by querying Google Scholar with
the terms "Inference Attacks on Federated Learning", "Membership Inference on
Federated Learning", "Model Inversion on Federated Learning", and "Gradient
Inversion on Federated Learning". The next section will cover the threats these
advances pose to current systems.

First, we will discuss various attacks, focusing not only on their performance
but paying special attention to the scenario in which the researchers placed the
hypothetical adversary. Then, we will cover defenses to some of these attacks.

## Attacking

Various types of attacks fall under the umbrella of inference attacks. As of the
writing of this essay, the most popular are _Membership Inference_ and _Gradient
Inversion_ as these show the most results. _Passive_ inference attacks are more
often covered than their _active_ counterparts. For each paper, we annotate
the type of attack (see [Inference Attacks](#inference-attacks)), summarize
the findings of the authors, and briefly discuss them.

### Do Gradient Inversion Attacks Make Federated Learning Unsafe?

Keywords: _Model Inversion_, _Local/Global_, _Cross-Device/HFL_, _Passive_

@hatamizadehGradientInversionAttacks2023 performed image reconstruction using
gradient inversion while relaxing a strong assumption made in prior work
regarding Batch Normalization (BN) [@ioffeBatchNormalizationAccelerating2015].
BN is a technique used in neural networks that significantly improve the
learning rate and stability, and is therefore ubiquitous in modern machine
learning. The technique introduces two learned parameters, $\beta$ and $\gamma$,
which thus change during the learning
process[@ioffeBatchNormalizationAccelerating2015]. Previous work has assumed
these statistics to be static [@geipingInvertingGradientsHow2020,
@kaissisEndtoendPrivacyPreserving2021], introducing an error that would compound
over time. The authors were able to reliably reconstruct images without assuming
static BN statistics. The authors make a strong case for an inversion attack
that could be used in practice but still rely on priors (approximations of the
image) to make accurate reconstructions.

### Improved Gradient Inversion Attacks and Defenses in Federated Learning

Keywords: _Membership Inference_, _Local/Global_, _Cross-Device/HFL_, _Passive_,
_White-Box_

@gengImprovedGradientInversion2023 proposed a framework for inverting both
_FedAVG_-based and _FedSGD_-based networks in an "honest-but-curious" scenario.
They mention prior work has failed to effectively perform gradient inversion
when FL  uses the _FedAVG_ aggregation algorithm. Furthermore, they specify
methods for fine-tuning the performance of image restoration in the inverted
model, allowing them to restore images that were introduced 16 epochs before the
current iteration. As Federated Learning is an iterative process, one can
imagine that the further a data point is removed from the current iteration, the
harder it is to infer from the current gradient. While their results are
promising, they do assume a white-box attack scenario making their attack harder
to perform.

### CS-MIA: Membership Inference Attack Based on Prediction Confidence Series in Federated Learning

Keywords: _Membership Inference_, _Local/Global_, _Cross-Device/VFL_, _Passive_

@guCSMIAMembershipInference2022 were able to determine whether data points are
members of certain datasets by following the trend in their classification
confidence. Over time, the global model should perform less well on
participant's private data, meaning that member data should follow a different
trend compared to non-member data. They then train a supervised model the
determine whether data points were part of the training set based on this
assumption. The model is then used to determine the probability of unseen data
being part of the target training data set. They show high accuracy and
F1-scores for all datasets with the lowest performer being MNIST (around 60%
compared to >90% for the other datasets). Still, the proposed solution scores
the best out of all included approaches by a significant margin.

### Subject Membership Inference Attacks in Federated Learning

Keywords: _Membership Inference_, _Local/Global_, _Cross-Silo/VFL_, _Passive_

In a black-box setting, @suriSubjectMembershipInference2022 propose a method
for what they call _Subject Inference_ (see [Federated
Learning](#federated-learning)). They describe previous work as being
disconnected from real-world scenarios as it (i) includes information
adversaries would not normally have access to and (ii) assumes the adversary is
looking for data points rather than individuals. Instead of determining whether
one particular data point was part of the training set, the authors attempt to
infer whether an individual, a _subject, (or rather their distribution) is
present in the dataset given some preexisting information on them. They show
the attack to be very effective in various real-world datasets while also
increasing the realism of the scenario. They show Subject Inference to be a
real threat to user privacy.

### Active Membership Inference Attack under Local Differential Privacy in Federated Learning.

Keywords: _Membership Inference_, _Local/Global_, _Cross-Device/HFL_, _Active_

Different from other works, @nguyenActiveMembershipInference2023 considers a
maximally malicious, i.e. _active_, membership inference attack. They implement
a method for inferring membership of a particular data point in the presence of
differential privacy [@dworkAlgorithmicFoundationsDifferential2013].
Differential privacy obscures the relation of the individual to the data point,
without affecting the patterns used for training machine learning models. The
authors show that their method performs well, even under such obscuring of the
data. Furthermore, the attack only starts to degrade after the level of
obscurity interferes with model performance. They show that more rigorous
privacy methods should be proposed to deal with such attacks.

## Defending

> - [ ] How can some of the aforementioned attacks be prevented or their risks
> reduced
> - [ ] What are the trade-offs, and do they make any implicit/explicit
> assumptions?
>
> $\Rightarrow$ We can **describe some tactics to mitigate the impact of the
> aforementioned attacks** and **use them in the context of a cost/benefit
> analysis**

To combat inference attacks, we discuss potential defenses against them. Some of
the papers that are included have been discussed in the last section. These have
been marked with a footnote accordingly [^1]. For each paper, we will summarize
the proposed measures and briefly discuss them.

### Improved Gradient Inversion Attacks and Defenses in Federated Learning[^1]

@gengImprovedGradientInversion2023 found that labels that only appeared only
once were more prone to their proposed inversion attacks (see
[](#improved-gradient-inversion-attacks-and-defenses-in-federated-learning)).
They also mention the use of larger batch sizes in the global model (i.e. more
clients) to reduce the amount of private information embedded in a single batch.
Lastly, they claim FedAVG possesses "stronger privacy preserving capabilities
than FedSGD". As this was included in the discussion of their attack-oriented
paper, they do not evaluate these claims further.

### Do Gradient Inversion Attacks Make Federated Learning Unsafe?[^1]

@hatamizadehGradientInversionAttacks2023 make several recommendations to make
existing implementations of FL safer, namely: (i) larger training sets, (ii)
updates from a larger number of iterations over different (iii) large batch
sizes. In addition, they mention three more changes that could potentially
mitigate server-side (i.e. _Global_) gradient inversion attacks: (1) The use of
_Homomorphic Encryption_ (see [Discussion](#discussion)), (2) ensuring the
attacker does not have knowledge of the model architecture, and (3) using an
alternative aggregation algorithm such as FedBN [@liFedBNFederatedLearning2021,
@andreuxSiloedFederatedLearning2020]. The countermeasures provided are
relatively general. They also provided sources affirming their suspicions.

### An Empirical Analysis of Image Augmentation Against Model Inversion Attack in Federated Learning

@shinEmpiricalAnalysisImage2023 propose the use of image augmentation as a
more viable alternative to differential privacy
[@dworkAlgorithmicFoundationsDifferential2013]. Image augmentation is a data
synthesis method that increases the size of the training set, and reduces
over-fitting [@shortenSurveyImageData2019]. As this introduces fake data while
improving the over-all performance of the model, the authors suggest it could be
used to mitigate model inversion attacks. They attack they used was introduced
by [@geipingInvertingGradientsHow2020], and various more successful attacks have
been constructed since then [@hatamizadehGradientInversionAttacks2023,@gengImprovedGradientInversion2023].

### ResSFL: A Resistance Transfer Framework for Defending Model Inversion Attack in Split Federated Learning

In a framework introduced by [@liResSFLResistanceTransfer2022], Split Federated
Learning (SFL) [@annavaramGroupKnowledgeTransfer] is augmented with an
discriminator-like model that attempts to invert the model before the client
sends their model to the central server. By choosing weights where the
discriminator performs poorly, they claim to improve the resiliency of the
scheme to model inversion attacks. They indeed show improvements over the
standard implementation of SFL, but do not mention how this method compares to
attacks on default FL.

[^1]: Often, novel attack proposals also include possible countermeasures. Some
of the papers covered in the last section, therefore, have also been included in
this section.

# Discussion

> - [ ] What can we say about the threat these attacks pose?
>   - [ ] How effective are these defenses?
>   - [ ] What is the trade-off?
> - [ ] What research can be done to mitigate these risks?
>
> $\Rightarrow$ We **understand what the current threat landscape for IA on FL
> looks like** and **know where to improve understanding to mitigate the
> risks**.

In this section, we will discuss the attacks and defenses as presented in the
last section. Specifically, we determine the threats these attacks pose and if
the defenses included could effectively mitigate the. In the end, we propose
new research directions that could help mitigate these threats.

## Current Threats and Trends

The attacks presented show how Federated Learning might not be able to guarantee
privacy. Privacy, thus, should still remain a concern even if Federated
Learning brings stronger privacy guarantees than traditional machine learning
and its derivatives. Let us summarize the threats these attacks pose:

- _More Realistic Scenarios_: Research starts to introduce more realistic
scenarios that could threaten current implementations of Federated Learning.  As
the field matures, attacks seem to become more realistic. Especially the work
presented by @suriSubjectMembershipInference2022 poses a real threat as it
assumes a complete black-box attack with reasonable assumptions while still
showing good performance. Even in complete black-box settings, however, we still
assume the ability to intercept and read the communications. Were this to be
encrypted, such attacks could possibly be mitigated
[@liPrivacyThreatsAnalysis2021].

- _Increased Resilience Against Existing Privacy Measures_: Some of the
aforementioned papers have shown improvements concerning the evasion of
privacy-preserving measures. @nguyenActiveMembershipInference2023 have shown
how a membership inference attack can be effectively performed in the presence
of differential privacy. Their method was effective to such a degree that the
attack was ineffective only once the privacy measures started to affect model
performance. The image augmentation countermeasure proposed by
[@shinEmpiricalAnalysisImage2023] could be a viable option. This countermeasure,
however, was only tested in a _passive_ scenario.

- _Stronger Attacks in Existing Scenarios_: As to be expected, some work was
focused on improving performance in existing scenarios.
@guCSMIAMembershipInference2022 have shown that much is still to be learned in
the field by proposing a relatively simple approach that improves upon all
previous methods by a large margin.

Such developments are not surprising, progress in both offense and defense is to
be expected. The speed at which research moves forward is very impressive and
suggests the field is still in early stages of development. When considering
using such new technologies in production, this could be considered when
assessing the security of such systems.

## Future Work

Considering the aforementioned advances, the following directions could provide
useful for future research:

1. Consider using existing preprocessing methods for privacy preservation.
@shinEmpiricalAnalysisImage2023 and @hatamizadehGradientInversionAttacks2023
both either use or suggest using existing pre-processing or other
learning-enhancing augmentations to improve privacy. Efforts toward generalizing
data _before_ training might prove a solution to both overfitting and privacy.

2. New attack methods would benefit from relaxing assumptions over instead
performance. Various of the attacks shown here have been able to relax
assumptions to a provide a more realistic assessment of the privacy-preserving
features of FL. While performance improvements might provide interesting result
and insights, focussing efforts on exposing potential _realistic_ threats would
have a more direct effect on our ability to asses FL from a privacy perspective.

# Conclusion

> - [ ] How 'risky' are the tactics described in this essay
> - [ ] Do the defenses have any real impact on the risk of these attacks
> - [ ] What would we need to research to improve any of these risks
>
> $\Rightarrow$ We can **describe the overall risk of model inversion attacks on
> federated learning** and **build on top of this paper to reduce their risks
> and improve our understanding**

This essay has provided an overview for security specialists and machine
learning specialists to assess the current state of Inference Attacks in
Federated Learning. Progress over the last year has shown the field to be
advancing quickly. Introducing successful attacks on new, more realistic
scenarios, showing the ability to circumvent mature privacy-preserving, measures
and improving the performance of existing methods.

# References
