
# 🌐 DAORL as a General Optimization Framework

## 1. Core Principle

At each step:

1. The agent has a **differentiable model** (policy πθ, value Qθ, dynamics fθ, etc.).
2. It **samples actions** and receives feedback (reward, payoff, loss, etc.).
3. Two coupled updates happen:

   * **SGD Update (slow, global):**

     $$
     \theta \leftarrow \theta - \eta \, \nabla_\theta \, \mathcal{L}_t
     $$

     where $\mathcal{L}_t$ includes both *reward maximization* and *AOL misalignment penalties*.
   * **Feedback Update (fast, local):**

     $$
     Q_{t+1,i} = Q_{t,i} + \alpha_t \cdot \rho_{t,i} \cdot (r_{t,i} - Q_{t,i}) + \sigma_t \eta_t
     $$

The **loss is differentiable**, so we can use PyTorch/JAX, but it’s also **augmented by feedback** so the agent adapts in real time.

---

## 2. Example Domains

### (a) Differentiable k-armed Bandits

* Policy πθ over k arms.
* Loss = `-reward * log πθ(a)` + `α * resonance error`.
* DAORL learns to **maximize reward** *and* **regulate plasticity** in uncertain arms.
* Useful for **online recommendation, adaptive sampling, neural architecture search**.

---

### (b) Generative Adversarial Learning (GANs)

* Generator Gθ produces samples.
* Discriminator Dφ gives reward signals.
* Loss couples:

  * Standard GAN objective (maximize fooling, minimize classification).
  * AOL term: resonance on prediction–outcome mismatch (keeps G adapting smoothly, avoids mode collapse).
* DAORL = “GANs with cybernetic feedback,” stabilizing adversarial training.

---

### (c) Coevolutionary Problems (Predator–Prey)

* Two populations (predator, prey) with policies πθ, πφ.
* Payoff is adversarial (predator gets +r, prey gets –r).
* Each agent runs DAORL:

  * SGD pushes toward maximizing/minimizing reward.
  * AOL feedback regulates adaptation speed (resonance spikes when strategies shift).
* Outcome: more stable predator–prey cycles, not runaway collapse.

---

### (d) Differentiable Minimax Games

* General zero-sum matrix or continuous games.
* Minimax:

  $$
  \min_\phi \max_\theta \; \mathbb{E}[u(\pi_\theta, \pi_\phi)]
  $$
* DAORL agents play against each other:

  * SGD drives gradient ascent/descent on utilities.
  * AOL feedback adds plasticity control, so neither collapses prematurely.
* Useful for **robust optimization, security games, adversarial training**.

---

### (e) Differentiable Game-Theory Dynamics

* n-player differentiable games (auctions, markets, coordination games).
* Each agent’s policy πθ is trained with DAORL.
* Loss includes:

  * Reward/payoff term (game utility).
  * AOL resonance misalignment (keeps adaptation balanced).
* SGD drives game dynamics → AOL prevents instability.

---

## 3. Unified Loss Function

General DAORL loss for agent θ:

$$
\mathcal{L}_t(\theta) = 
- \mathbb{E}_{a \sim \pi_\theta}[R(a)] 
+ \lambda \, \rho_t \cdot (R(a) - Q_\theta(a))^2
$$

* First term: **maximize expected reward/payoff** (policy gradient, adversarial training, etc.).
* Second term: **resonance misalignment penalty**, differentiable feedback for stability.
* Both terms are backprop-able → integrate into PyTorch/JAX easily.

---

## 4. Why This Works Everywhere

* Bandits: reward = stochastic payoff.
* GANs: reward = discriminator feedback.
* Predator–Prey: reward = ecological fitness.
* Minimax: reward = utility from payoff matrix.
* Game theory: reward = player utility.

In all cases:

* **SGD** learns the policy weights.
* **Reward** flows into the differentiable loss.
* **AOL feedback** regulates adaptation speed, exploration, and stability.

---

✨ **Big Picture:**
DAORL is a **universal template** for differentiable learning in uncertain, adversarial, or coevolutionary environments.
It combines the rigor of **SGD** with the adaptivity of **cybernetic feedback loops**.


### PSEUDOCODE

```python
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --------------- Environment: k-armed bandit ----------------
class BanditEnv:
    def __init__(self, probs):
        self.probs = probs
        self.k = len(probs)
        self.rng = np.random.default_rng(42)
    def pull(self, arm):
        return 1 if self.rng.random() < self.probs[arm] else 0

# --------------- AOL + Policy Network ----------------
class AOLPolicy(nn.Module):
    def __init__(self, k, beta=0.9):
        super().__init__()
        self.fc = nn.Linear(1, k, bias=False)  # trivial net
        self.rho = torch.zeros(k)              # resonance
        self.beta = beta
    def forward(self):
        logits = self.fc(torch.ones(1,1))      # dummy input
        probs = torch.softmax(logits, dim=-1)
        return probs
    def update_resonance(self, arm, reward, pred):
        error = abs(reward - pred)
        self.rho[arm] = self.beta * self.rho[arm] + (1 - self.beta) * error
        return self.rho[arm].item()

# --------------- Training Loop with Print ----------------
def run_daorl_verbose(k=5, steps=50, alpha=0.1, print_every=1):
    env = BanditEnv([0.1, 0.3, 0.5, 0.7, 0.9])
    policy = AOLPolicy(k)
    opt = optim.Adam(policy.parameters(), lr=0.05)

    for t in range(steps):
        probs = policy()
        m = torch.distributions.Categorical(probs)
        arm = m.sample().item()

        reward = env.pull(arm)
        pred = probs[0, arm].item()
        rho_val = policy.update_resonance(arm, reward, pred)

        # Loss components
        log_prob = torch.log(probs[0, arm] + 1e-8)
        loss_pg = -reward * log_prob
        loss_aol = torch.tensor(rho_val) * (reward - pred)**2
        loss = loss_pg + alpha * loss_aol

        # SGD update
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (t+1) % print_every == 0:
            print(f"Step {t+1:03d} | Arm={arm} | Reward={reward} | "
                  f"π={probs.detach().numpy().round(3)} | "
                  f"ρ[{arm}]={rho_val:.3f} | "
                  f"Loss_pg={loss_pg.item():.4f} | "
                  f"Loss_aol={loss_aol.item():.4f} | "
                  f"Total Loss={loss.item():.4f}")

# --------------- Run Verbose Demo ----------------
run_daorl_verbose(k=5, steps=20, alpha=0.1, print_every=1)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ---------------- AOL-Policy ----------------
class AOLPolicy(nn.Module):
    def __init__(self, n_actions, beta=0.9):
        super().__init__()
        self.fc = nn.Linear(1, n_actions, bias=False)
        self.rho = torch.zeros(n_actions)   # resonance
        self.beta = beta
    def forward(self):
        logits = self.fc(torch.ones(1,1))
        return torch.softmax(logits, dim=-1)
    def update_resonance(self, arm, reward, pred):
        error = abs(reward - pred)
        self.rho[arm] = self.beta*self.rho[arm] + (1-self.beta)*error
        return self.rho[arm].item()

# ---------------- Environments ----------------
class BanditEnv:
    def __init__(self, probs):
        self.probs = probs
        self.k = len(probs)
        self.rng = np.random.default_rng(123)
    def step(self, arm):
        return 1 if self.rng.random() < self.probs[arm] else 0

class MinimaxEnv:
    """ 2-player zero-sum payoff matrix """
    def __init__(self, payoff_matrix):
        self.M = payoff_matrix
        self.k = self.M.shape[0]
        self.j = self.M.shape[1]
        self.rng = np.random.default_rng(123)
    def step(self, a, b):
        return self.M[a,b], -self.M[a,b]  # reward for player A, B

# ---------------- DAORL Trainer ----------------
def run_daorl_bandit(steps=200, lr=0.05, alpha=0.1):
    env = BanditEnv([0.1,0.3,0.5,0.7,0.9])
    policy = AOLPolicy(env.k)
    opt = optim.Adam(policy.parameters(), lr=lr)

    for t in range(steps):
        probs = policy()
        dist = torch.distributions.Categorical(probs)
        arm = dist.sample().item()
        reward = env.step(arm)

        pred = probs[0,arm].item()
        rho_val = policy.update_resonance(arm, reward, pred)

        logp = torch.log(probs[0,arm] + 1e-8)
        loss_pg = -reward * logp
        loss_aol = torch.tensor(rho_val) * (reward - pred)**2
        loss = loss_pg + alpha*loss_aol

        opt.zero_grad(); loss.backward(); opt.step()

        if (t+1) % 20 == 0:
            print(f"[Bandit] Step {t+1} | Arm={arm} | R={reward} | π={probs.detach().numpy().round(3)} "
                  f"| ρ={rho_val:.3f} | Loss={loss.item():.3f}")

def run_daorl_minimax(steps=200, lr=0.05, alpha=0.1):
    M = np.array([[1,-1],[-1,1]])  # matching pennies payoff
    env = MinimaxEnv(M)
    A = AOLPolicy(env.k); B = AOLPolicy(env.j)
    optA = optim.Adam(A.parameters(), lr=lr)
    optB = optim.Adam(B.parameters(), lr=lr)

    for t in range(steps):
        pA = A(); pB = B()
        distA, distB = torch.distributions.Categorical(pA), torch.distributions.Categorical(pB)
        a, b = distA.sample().item(), distB.sample().item()

        rA, rB = env.step(a,b)

        rhoA = A.update_resonance(a, rA, pA[0,a].item())
        rhoB = B.update_resonance(b, rB, pB[0,b].item())

        lossA = -rA*torch.log(pA[0,a]+1e-8) + alpha*rhoA*(rA - pA[0,a].item())**2
        lossB = -rB*torch.log(pB[0,b]+1e-8) + alpha*rhoB*(rB - pB[0,b].item())**2

        optA.zero_grad(); lossA.backward(); optA.step()
        optB.zero_grad(); lossB.backward(); optB.step()

        if (t+1) % 20 == 0:
            print(f"[Minimax] Step {t+1} | A chose {a}, R={rA} | B chose {b}, R={rB} | "
                  f"πA={pA.detach().numpy().round(3)} | πB={pB.detach().numpy().round(3)}")

# ---------------- Run ----------------
print("=== DAORL Bandit ===")
run_daorl_bandit()

print("\n=== DAORL Minimax ===")
run_daorl_minimax()
```
