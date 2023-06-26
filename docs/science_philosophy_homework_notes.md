---
title: Seminar Cyber Security - Homework Assignment
author: Zohar Cochavi (4962281)
geometry: margin=1in
---

# Question 1

_Come up with a problem or research question related to cyber security
that you think could be solved or answered by relying on or bringing
in:_

a. _exclusively knowledge and methods from science, i.e. one or more of the
natural sciences_

> **What is the impact of ethical and safety measures on LLM performance?**
>
> As long as we define performance in a strict sense, the information necessary to
> conclude rests solely upon the measurements made during the experiments.

b. _exclusively knowledge and methods from one or more of the social sciences_

> **Why are security policies enforced in the workforce rarely followed to their
> fullest extent?**
>
> This only requires interviewing employees on their experience with
> security policies, and their motivation for not following them. No
> measurements, in the empirical sense, are required.

c. _exclusively knowledge and methods from mathematics and/or theoretical
computer science (algorithmics, a.s.f.)_

> **How can AES and other modular exponentiation-based encryption algorithms be
> effectively attacked by quantum computing algorithms.**
>
> Investigating this exclusively requires knowledge about quantum algorithms and
> classical encryption methods. Here, we use effectively in the sense that it
> has a large reduction in computational complexity over classical methods.

# Question 2

a. _Do you think that the term ‘social engineering’ as used by hackers is correct
in that these techniques can rightly be seen as a form of engineering?_

> Yes. Social engineering could be described as the application of sociological
> principles to exploit human behavior. As we are talking about an application to
> achieve a goal, it is reasonable to assume social engineering consists at least
> mostly of claims that are normative in nature. One might argue that because
> social engineering rests on sociological principles and claims, it cannot be
> ruled out that it is a science. The instructions that social engineering produces,
> however, are _based_ on sociology. So, while it _uses_ descriptive claims, it
> does not _make_ any descriptive claims. The field _uses_ science, but is not a
> science. Furthermore, as long as we assume that the normative claims made in
> social engineering are indeed based on science, and thus empirical measurements,
> we can also exclude it as a non-science. Assuming, we can either categorize the
> field as _engineering_, _science_, or _non-science_, we can conclude that it is
> rightfully called social _engineering_.

b. _Do you think that the use of ‘engineering’ by Popper in ‘piecemeal social
engineering’ IS rightly chosen, in that (i) to organize society, and (ii) to
modify and adapt the organization of society are basically engineering
problems?_

> In the same sense as before, these problems be called engineering problems,
> given that they are still based on scientific research. To organize and to
> modify are action-oriented tasks and they require insight into what might be
> potential good actions (hence the basis in empirical evidence, i.e. science).
> More certainty than potentially good actions, however, cannot be guaranteed,
> making them more akin to recommendations than a truth or falsity. They might
> not be considered 'classical' engineering problems, but I see not good reason
> not the consider them as such.

# Question 3

a. _What makes all three paradigms fundamentally different? Choose
to answer this from the perspective of the methodology or the
epistemology._

> The epistemological perspective tells us that the difference between the three
> paradigms is in the existence, or availability of _a priori_ and _a
> posteriori_ knowledge. In other words whether we can have knowledge about what
> a program _will do_ and what it _has done_. In the **rationalist** paradigm,
> there is the assumption that full knowledge of the program is available before
> executing it. The **technocratic** paradigm says that knowledge about a
> computer program can only be gathered by experience, by testing. Finally, the
> **scientific** paradigm says that both are available, but the _a priori_
> knowledge might be limited.

b. _What objection would a partisan of the technocratic paradigm present to an
advocate of the rationalist paradigm?_

> Computer programs built in production are simply too complex to approach from
> a rationalist view. The sheer amount of different paths a modern application
> can take is more than any theoretical analysis could cover. Therefore, the
> only analysis that would be practical would be to test the program by
> executing it.

c. _What response would the latter have to justify his/her own paradigm?_

> The reality is, however, that all the information about the program is
> available. The program is itself a full description of its execution. Whether
> all of it can be practically analyzed is not relevant.

# Question 4

a. _What chief implication for scientific and engineering research follows from
the kind of inconclusive evidence characteristic of algorithms? Why?_

**That results provided by algorithms must be carefully investigated before
conclusions are drawn from them.**

**Algorithms are great at analyzing large amounts of data, but they are often
based on large, complex computational models.**

> From the epistemic perspective, this is problematic because it results in a
> lack of transparency with regard to the collection of the evidence. We thus
> end up with evidence that is inconclusive and inscrutable. Finally, because of
> biases in data (which are a result of biases in human behavior) the model will
> inevitable also display biased behavior, and thus produce potentially
> misguided evidence.
>
> From the normative perspective, _unfair outcomes_, _transformative effects_
> and _traceability_ are at odds when using evidence provided by algorithms.
> The biased behavior mentioned before could result in consistent unfair
> outcomes which could in turn lead to discrimination. This in turn would have
> transformative effects. Finally, the tracability of harm (i.e. who is
> conclusively responsible for harm caused by an algorithm) is limited, as the
> responsibility of the effect is distributed.
>
> In all cases, the problem arises either from a lack of interpretability or an
> increase of scale.

b. _Briefly describe the relation between reliable results, black-boxes, and
unfair outcomes. Offer an example._

> Black-box models are inherently hard to interpret, as their inner workings,
> and therefore their reasoning, are unknown. A strong argument for the
> reliability of results is by proving the model is reliable. This does,
> however, require insight into the model. As this cannot be done for black-box
> models, we can only determine the reliability of observed outputs and from
> there induce the reliability of the results in general. The less reliable a
> system is, the more unfair it is.

c. _Which transformative effects do algorithms have in our personal,
professional, or academic life? Why?_

> Algorithms have an incredible, tangible impact on daily life. They provide
> personal entertainment (reccomendation algorithms), reduce overhead for
> interacting with other software (voice commands, virtual assistants), stimulate
> creative output (ChatGPT). The sheer amount of people currently using ChatGPT to
> aid in writing is enough to stimulate almost every university to issue a
> statement on their opinion on the use of ChatGPT in relation to academic work.

d. _How could the accountability gap (i.e., ‘causality’, ‘justice’ and
‘compensation’) be reduced or fully eliminated? Focus on one of the three and
elaborate._

> Most AI-based applications are created by larger corporations. Focusing on the
> issue of 'causality', let's take the self-driving cars Tesla provides as an
> example[^1]. Instead of looking at the company as a _technology company that
> manufactures self-driving cars_, we should look at the company as a _car
> manufacturer that allows for self-driving_ and a _chauffeur service_.
> This changes the company from 'a company that produces a self-driving AI' to
> 'a company that provides a private chauffeur service
> _through_ an AI'. The latter implies that we can hold the company accountable
> for any harms done in a similar manner as we would, for example, a personal
> chauffeur. The requirement, of course, is that the car is in a state where it is
> safe to drive. If an individual were to never replace their disk brakes, it
> would be hard to hold the private chauffeur accountable for an accident that was
> caused by faulty brakes. If, however, it has been shown that for this particular
> car the disk brakes have been shown to be degrading exceptionally fast, we
> could, hold the car manufacturer accountable[^2].

[^1]: The company only provides full self-driving in beta, but this is only
    supposed to be an example. For this discussion, we assume Tesla provides
    fully self-driving cars.

[^2]: I realize this answer is technically over the word limit, but I had a lot
    of fun with it. I hope the shorter length of the other sub-questions
    compensates for this.
