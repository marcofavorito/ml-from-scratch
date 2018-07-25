# Reinforcement Learning

## Dynamic systems

### Taxonomy

Type of evolutions:
- Deterministic
- Non-Deterministic
- Probabilistic (Stochastic)

Representation of states:
- Explicit
- Implicit

Characteristics of the history of temporal evolution:
- Markov
- Semi-Markov
- Non-Markov

Markov property:
> evolution of the system does not depend on history

Characteristics of the observations
- Full observability
- Partial observability


### Representation

    X set of states
    A set of actions
    d transition function
        - deterministic
            X x A -> X
        - not-deterministic
            X x A -> 2^X
        - probabilistic
            X x A x X -> P(x'|x,a)
    Z set of observations (discrete, continuous, probabilistic)
    
### Common models

- Markov Decision Process MDP
    - states fully observable
- Hidden Markov Model HMM
    - states not observable directly
    - Observation uncertainty modeled as P(z|x,a)
- Partially Observable Markov Decision Process POMDP
    - partial observability with control action
    - P(z|x)

They are all Dyamic Bayesian Networks, generalized in Probabilistic Graphical Models (PGM)


### MDP

- Deterministic transitions:
    - reward function: X x A -> R (or X -> R)
- Non-deterministic transitions:
    - reward function: X x A x X -> R
- Stochastic transitions:
    - reward function: X x A x X -> R
    

#### Solution: optimal policy
Policy:
    
    pi: X -> A
    
Optimality: maximize cumulative reward

if MDP completely known     => reasoning/planning
if MDP not completely known => learning

Equivalently: maximize Expected value of cumulative discounted reward

    V^pi(x) = E\[r1 + g*r2 + g^2*r3 + ...\]

optimal policy: 
    
    argmax_pi V(x) 
    
Deterministic case:

    V_t(x) = r_t + g*V_(t+1)(x)

Non-deterministic: expected value.

Q-function:
Expected value of executing a in x, and then act according to pi:

    Q^pi(x,a) = sum_X P(x'|x,a)*(r(x,a,x') + g*V^pi(x))
    

    V^pi(x) = Q^pi(x, pi(x))
    

Solutions:
- Reasoning
- Value Iteration
- Policy Iteration