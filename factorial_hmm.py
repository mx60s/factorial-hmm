{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mx60s/factorial-hmm/blob/main/factorial_hmm_clean.ipynb)\n\n# Factorial HMM: Belief Geometry and Explaining Away\n\nI went looking for HMMs which could satisfy some conditions for the project: ergodic, naturalistic for LLMs, and requires \"representations to track structures more elaborate than square and ring graphs\". To that end I found this paper: https://www.ee.columbia.edu/~sfchang/course/svia-F03/papers/factorial-HMM-97.pdf\n\nThe coolest part of the paper is the modelling of Bach's chorales but I think factorial HMMs could be implicated in LLM pretraining in quite a few different ways, so it's specific enough to find structures but vague enough to be expanded upon for future work.\n\nA factorial HMM has multiple independent Markov chains that interact only through shared observations. Chains that are independent in the prior become coupled in the posterior and this is called \"explaining away\".\n\nHere's a simple model:\n- **Chain 1 (Formality)**: {Formal, Informal}\n- **Chain 2 (Topic)**: {Technical, Casual}\n- **Joint states**: (F,T), (F,C), (I,T), (I,C)"
  },
  {
   "cell_type": "code",
   "source": "# Colab setup\nimport sys\nif 'google.colab' in sys.modules:\n    !pip install -q plotly\n    \nimport numpy as np\nimport plotly.graph_objects as go\nfrom plotly.subplots import make_subplots\n\nclass FactorialHMM:\n    def __init__(self, vocab_size='8-token'):\n        self.A1 = np.array([[0.8, 0.2], [0.3, 0.7]])\n        self.A2 = np.array([[0.7, 0.3], [0.4, 0.6]])\n        self.A_joint = np.kron(self.A1, self.A2)\n        self.pi = np.array([0.25, 0.25, 0.25, 0.25])\n        self.state_names = ['(F,T)', '(F,C)', '(I,T)', '(I,C)']\n        self.vocab_size = vocab_size\n        if vocab_size == '8-token':\n            self.token_names = ['shall', 'gonna', 'algorithm', 'stuff', 'the', 'optimize', 'hey', 'pursuant']\n            self.B = np.array([\n                [0.15, 0.02, 0.25, 0.02, 0.20, 0.20, 0.01, 0.15],\n                [0.20, 0.05, 0.05, 0.10, 0.30, 0.05, 0.05, 0.20],\n                [0.02, 0.15, 0.25, 0.10, 0.20, 0.18, 0.05, 0.05],\n                [0.02, 0.25, 0.03, 0.20, 0.20, 0.02, 0.20, 0.08],\n            ])\n        else:\n            self.token_names = ['formal_tech', 'formal_casual', 'informal_tech', 'informal_casual']\n            self.B = np.array([\n                [0.70, 0.15, 0.10, 0.05],\n                [0.15, 0.70, 0.05, 0.10],\n                [0.10, 0.05, 0.70, 0.15],\n                [0.05, 0.10, 0.15, 0.70],\n            ])\n\n    def sample(self, n_steps, seed=None):\n        if seed is not None:\n            np.random.seed(seed)\n        states = np.zeros(n_steps, dtype=int)\n        observations = np.zeros(n_steps, dtype=int)\n        states[0] = np.random.choice(4, p=self.pi)\n        observations[0] = np.random.choice(len(self.B[0]), p=self.B[states[0]])\n        for t in range(1, n_steps):\n            states[t] = np.random.choice(4, p=self.A_joint[states[t-1]])\n            observations[t] = np.random.choice(len(self.B[0]), p=self.B[states[t]])\n        return states, observations\n\n    def forward(self, observations):\n        T = len(observations)\n        beliefs = np.zeros((T, 4))\n        alpha = self.pi * self.B[:, observations[0]]\n        alpha = alpha / alpha.sum()\n        beliefs[0] = alpha\n        for t in range(1, T):\n            alpha_pred = self.A_joint.T @ alpha\n            alpha = alpha_pred * self.B[:, observations[t]]\n            alpha = alpha / alpha.sum()\n            beliefs[t] = alpha\n        return beliefs\n\n    def compute_coupling(self, joint_beliefs):\n        T = len(joint_beliefs)\n        coupling = np.zeros(T)\n        for t in range(T):\n            p = joint_beliefs[t]\n            p_f, p_i = p[0]+p[1], p[2]+p[3]\n            p_t, p_c = p[0]+p[2], p[1]+p[3]\n            p_indep = np.array([p_f*p_t, p_f*p_c, p_i*p_t, p_i*p_c])\n            coupling[t] = np.linalg.norm(p - p_indep)\n        return coupling\n\nhmm = FactorialHMM(vocab_size='8-token')",
   "metadata": {},
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Chain 1 (Formality):        Chain 2 (Topic):\n",
      "         F     I                   T     C\n",
      "    F [0.8   0.2]         T [0.7   0.3]\n",
      "    I [0.3   0.7]         C [0.4   0.6]\n"
     ]
    }
   ],
   "source": [
    "# transition matrices\n",
    "print(\"Chain 1 (Formality):        Chain 2 (Topic):\")\n",
    "print(\"         F     I                   T     C\")\n",
    "print(f\"    F [{hmm.A1[0,0]:.1f}   {hmm.A1[0,1]:.1f}]         T [{hmm.A2[0,0]:.1f}   {hmm.A2[0,1]:.1f}]\")\n",
    "print(f\"    I [{hmm.A1[1,0]:.1f}   {hmm.A1[1,1]:.1f}]         C [{hmm.A2[1,0]:.1f}   {hmm.A2[1,1]:.1f}]\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## The Belief Simplex\n",
    "\n",
    "Beliefs over 4 states live on a 3-simplex (tetrahedron). The **product manifold** is where P(S₁,S₂) = P(S₁)×P(S₂) — i.e., where chains are independent.\n",
    "\n",
    "- Prior dynamics preserve products: if you start on the manifold, you stay on it\n",
    "- Observations make the posterior move off the manifold"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Tetrahedron vertices for 4-state simplex\n",
    "VERTICES = np.array([[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]) / np.sqrt(3)\n",
    "STATE_NAMES = ['(F,T)', '(F,C)', '(I,T)', '(I,C)']\n",
    "\n",
    "def belief_to_3d(belief):\n",
    "    return belief @ VERTICES\n",
    "\n",
    "def create_simplex_figure():\n",
    "    \"\"\"Create interactive 3D simplex with product manifold.\"\"\"\n",
    "    fig = go.Figure()\n",
    "    \n",
    "    # Tetrahedron edges\n",
    "    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]\n",
    "    for i, j in edges:\n",
    "        fig.add_trace(go.Scatter3d(\n",
    "            x=[VERTICES[i,0], VERTICES[j,0]], \n",
    "            y=[VERTICES[i,1], VERTICES[j,1]], \n",
    "            z=[VERTICES[i,2], VERTICES[j,2]],\n",
    "            mode='lines', line=dict(color='gray', width=2), \n",
    "            showlegend=False, hoverinfo='skip'\n",
    "        ))\n",
    "    \n",
    "    # Vertex labels\n",
    "    fig.add_trace(go.Scatter3d(\n",
    "        x=VERTICES[:,0]*1.15, y=VERTICES[:,1]*1.15, z=VERTICES[:,2]*1.15,\n",
    "        mode='text', text=STATE_NAMES, textfont=dict(size=12),\n",
    "        showlegend=False, hoverinfo='skip'\n",
    "    ))\n",
    "    \n",
    "    # Product manifold surface\n",
    "    n = 20\n",
    "    p1 = np.linspace(0.01, 0.99, n)\n",
    "    p2 = np.linspace(0.01, 0.99, n)\n",
    "    X, Y, Z = np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))\n",
    "    for i, pf in enumerate(p1):\n",
    "        for j, pt in enumerate(p2):\n",
    "            joint = np.array([pf*pt, pf*(1-pt), (1-pf)*pt, (1-pf)*(1-pt)])\n",
    "            pt3d = belief_to_3d(joint)\n",
    "            X[i,j], Y[i,j], Z[i,j] = pt3d\n",
    "    \n",
    "    fig.add_trace(go.Surface(\n",
    "        x=X, y=Y, z=Z, opacity=0.3, colorscale=[[0,'green'],[1,'green']],\n",
    "        showscale=False, name='Product Manifold',\n",
    "        hovertemplate='Product Manifold<extra></extra>'\n",
    "    ))\n",
    "    \n",
    "    fig.update_layout(\n",
    "        scene=dict(xaxis_title='', yaxis_title='', zaxis_title='',\n",
    "                   aspectmode='cube'),\n",
    "        margin=dict(l=0, r=0, t=30, b=0), height=500\n",
    "    )\n",
    "    return fig"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Prior vs Posterior Trajectories\n",
    "\n",
    "**Prior** (green): Evolve beliefs using only transition dynamics.\n",
    "**Posterior** (blue): Update beliefs with actual observations.\n",
    "\n",
    "The prior trajectory stays exactly on the product manifold (coupling = 0).  \n",
    "The posterior trajectory moves off it (coupling > 0) — this is explaining away."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'hmm' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mNameError\u001b[39m                                 Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[1]\u001b[39m\u001b[32m, line 2\u001b[39m\n\u001b[32m      1\u001b[39m n_steps = \u001b[32m40\u001b[39m\n\u001b[32m----> \u001b[39m\u001b[32m2\u001b[39m states, observations = \u001b[43mhmm\u001b[49m.sample(n_steps, seed=\u001b[32m42\u001b[39m)\n\u001b[32m      4\u001b[39m posterior = hmm.forward(observations)\n\u001b[32m      6\u001b[39m prior = np.zeros((n_steps, \u001b[32m4\u001b[39m))\n",
      "\u001b[31mNameError\u001b[39m: name 'hmm' is not defined"
     ]
    }
   ],
   "source": [
    "n_steps = 40\n",
    "states, observations = hmm.sample(n_steps, seed=42)\n",
    "\n",
    "posterior = hmm.forward(observations)\n",
    "\n",
    "prior = np.zeros((n_steps, 4))\n",
    "belief = hmm.pi.copy()\n",
    "for t in range(n_steps):\n",
    "    prior[t] = belief\n",
    "    belief = hmm.A_joint.T @ belief\n",
    "\n",
    "prior_coupling = hmm.compute_coupling(prior)\n",
    "posterior_coupling = hmm.compute_coupling(posterior)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "hoverinfo": "skip",
         "line": {
          "color": "gray",
          "width": 2
         },
         "mode": "lines",
         "showlegend": false,
         "type": "scatter3d",
         "x": [
          0.5773502691896258,
          0.5773502691896258
         ],
         "y": [
          0.5773502691896258,
          -0.5773502691896258
         ],
         "z": [
          0.5773502691896258,
          -0.5773502691896258
         ]
        },
        {
         "hoverinfo": "skip",
         "line": {
          "color": "gray",
          "width": 2
         },
         "mode": "lines",
         "showlegend": false,
         "type": "scatter3d",
         "x": [
          0.5773502691896258,
          -0.5773502691896258
         ],
         "y": [
          0.5773502691896258,
          0.5773502691896258
         ],
         "z": [
          0.5773502691896258,
          -0.5773502691896258
         ]
        },
        {
         "hoverinfo": "skip",
         "line": {
          "color": "gray",
          "width": 2
         },
         "mode": "lines",
         "showlegend": false,
         "type": "scatter3d",
         "x": [
          0.5773502691896258,
          -0.5773502691896258
         ],
         "y": [
          0.5773502691896258,
          -0.5773502691896258
         ],
         "z": [
          0.5773502691896258,
          0.5773502691896258
         ]
        },
        {
         "hoverinfo": "skip",
         "line": {
          "color": "gray",
          "width": 2
         },
         "mode": "lines",
         "showlegend": false,
         "type": "scatter3d",
         "x": [
          0.5773502691896258,
          -0.5773502691896258
         ],
         "y": [
          -0.5773502691896258,
          0.5773502691896258
         ],
         "z": [
          -0.5773502691896258,
          -0.5773502691896258
         ]
        },
        {
         "hoverinfo": "skip",
         "line": {
          "color": "gray",
          "width": 2
         },
         "mode": "lines",
         "showlegend": false,
         "type": "scatter3d",
         "x": [
          0.5773502691896258,
          -0.5773502691896258
         ],
         "y": [
          -0.5773502691896258,
          -0.5773502691896258
         ],
         "z": [
          -0.5773502691896258,
          0.5773502691896258
         ]
        },
        {
         "hoverinfo": "skip",
         "line": {
          "color": "gray",
          "width": 2
         },
         "mode": "lines",
         "showlegend": false,
         "type": "scatter3d",
         "x": [
          -0.5773502691896258,
          -0.5773502691896258
         ],
         "y": [
          0.5773502691896258,
          -0.5773502691896258
         ],
         "z": [
          -0.5773502691896258,
          0.5773502691896258
         ]
        },
        {
         "hoverinfo": "skip",
         "mode": "text",
         "showlegend": false,
         "text": [
          "(F,T)",
          "(F,C)",
          "(I,T)",
          "(I,C)"
         ],
         "textfont": {
          "size": 12
         },
         "type": "scatter3d",
         "x": {
          "bdata": "YdRl9hk/5T9h1GX2GT/lP2HUZfYZP+W/YdRl9hk/5b8=",
          "dtype": "f8"
         },
         "y": {
          "bdata": "YdRl9hk/5T9h1GX2GT/lv2HUZfYZP+U/YdRl9hk/5b8=",
          "dtype": "f8"
         },
         "z": {
          "bdata": "YdRl9hk/5T9h1GX2GT/lv2HUZfYZP+W/YdRl9hk/5T8=",
          "dtype": "f8"
         }
        },
        {
         "colorscale": [
          [
           0,
           "green"
          ],
          [
           1,
           "green"
          ]
         ],
         "hovertemplate": "Product Manifold<extra></extra>",
         "name": "Product Manifold",
         "opacity": 0.3,
         "showscale": false,
         "type": "surface",
         "x": {
          "bdata": "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",
          "dtype": "f8",
          "shape": "20, 20"
         },
         "y": {
          "bdata": "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",
          "dtype": "f8",
          "shape": "20, 20"
         },
         "z": {
          "bdata": "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",
          "dtype": "f8",
          "shape": "20, 20"
         }
        },
        {
         "line": {
          "color": "green",
          "width": 4
         },
         "marker": {
          "color": "darkgreen",
          "size": 3
         },
         "mode": "lines+markers",
         "name": "Prior (no obs)",
         "type": "scatter3d",
         "x": {
          "bdata": "AAAAAAAAAAAsuOYIco+tPyEKrYaVK7Y/KOHJx4PduT+sTFjoera7P26Cn3j2orw/TR3DQDQZvT++6tQkU1S9P3TR3Zbicb0/zkTiT6qAvT9+fmQsDoi9P1SbpRrAi70/wCnGEZmNvT/2cFaNhY69P5CUHsv7jr0/YKYC6jaPvT9Dr3R5VI+9P7SzLUFjj70/7DUKpWqPvT8Kd/hWbo+9P5iX7y9wj70/4CdrHHGPvT8D8KiScY+9PxTUx81xj70/HEZX63GPvT8g/x76cY+9P6LbggFyj70/5ck0BXKPvT8GwQ0Hco+9P5Y8+gdyj70/W3pwCHKPvT9CmasIco+9P7IoyQhyj70/avDXCHKPvT9GVN8Ico+9PzUG4whyj70/K9/kCHKPvT+ly+UIco+9P+VB5ghyj70/An3mCHKPvT8=",
          "dtype": "f8"
         },
         "y": {
          "bdata": "AAAAAAAAAAAwuOYIco+tP1CRL+zWNrM/rgfo5F+LtD+YkVL8iPG0P5BUv+kuELU/DI/5sGAZtT+YoCTTIhy1P0CMMar2HLU/WjmCNzYdtT98oE1ISR21P9SligBPHbU/bifQt1AdtT+2NJg7UR21P4IFIWNRHbU/IkT9blEdtT+2I4xyUR21P8pmnXNRHbU/UGHvc1EdtT9E+Qd0UR21PwxaD3RRHbU/rJARdFEdtT+wOhJ0UR21P6xtEnRRHbU/+HwSdFEdtT+QgRJ0UR21P/aCEnRRHbU/XIMSdFEdtT94gxJ0UR21P4KDEnRRHbU/hoMSdFEdtT+GgxJ0UR21P4KDEnRRHbU/goMSdFEdtT+EgxJ0UR21P4SDEnRRHbU/hoMSdFEdtT+CgxJ0UR21P4KDEnRRHbU/gIMSdFEdtT8=",
          "dtype": "f8"
         },
         "z": {
          "bdata": "AAAAAAAAAADMxuvT9KV3P2Cu0oGbDoc/VKQR2h/DjD9w2nt6TWqPP1iuWijxUpA/1FORvpWdkD9+3zFxiMGQP8h7n48U05A/aJPrSbnbkD+ACpaRAeCQPw6qu6si4pA/lrsDTzLjkD8G4HHaueOQP3D6FYv945A/ih4VXR/kkD8c9C5EMOSQP/4jKrc45JA/dgN88DzkkD9w1RcNP+SQPwbPYRtA5JA/np2FokDkkD9UKhfmQOSQP27V3wdB5JA/0iLEGEHkkD8gRzYhQeSQP4BYbyVB5JA/9OCLJ0HkkD8aJZooQeSQPy5HISlB5JA/PNhkKUHkkD+6oIYpQeSQP/iElylB5JA/EvefKUHkkD82MKQpQeSQP6xMpilB5JA//FqnKUHkkD8c4qcpQeSQP6QlqClB5JA/akeoKUHkkD8=",
          "dtype": "f8"
         }
        },
        {
         "line": {
          "color": "blue",
          "width": 2
         },
         "marker": {
          "color": {
           "bdata": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJw==",
           "dtype": "i1"
          },
          "colorbar": {
           "len": 0.5,
           "title": {
            "text": "Time"
           },
           "x": 1
          },
          "colorscale": [
           [
            0,
            "rgb(247,251,255)"
           ],
           [
            0.125,
            "rgb(222,235,247)"
           ],
           [
            0.25,
            "rgb(198,219,239)"
           ],
           [
            0.375,
            "rgb(158,202,225)"
           ],
           [
            0.5,
            "rgb(107,174,214)"
           ],
           [
            0.625,
            "rgb(66,146,198)"
           ],
           [
            0.75,
            "rgb(33,113,181)"
           ],
           [
            0.875,
            "rgb(8,81,156)"
           ],
           [
            1,
            "rgb(8,48,107)"
           ]
          ],
          "showscale": true,
          "size": 4
         },
         "mode": "lines+markers",
         "name": "Posterior (with obs)",
         "type": "scatter3d",
         "x": {
          "bdata": "2i7E/4Pv0D/4rggnTQ/PP20OoH+JldK/WohEYXWdyD8nyiEVRPfKP6AtP2tBl9g/jKiCmCrY0D9+rxKyOTfJP6rDQWRXDsk/928KgWwGxT9+dhDJvunfP7OsjPtnQtQ/UhokGX9T0D8gnqfo3arMPxO+Hnm/QuA/fWAV0+Q84T8no3p1Ux7eP6lpvUM9Sbe/7JUHgEBi2b/Ch57QQmLWv0/qBeZhmKq/Z8l2oums0j8WapUMnF/QPzbwikpa9sw/H1iTY8P2xj+Xtpy5A8/Pv5ZQtt4FCtm/vkNjTbYAwj+yGfTxUP/CP9mQDxol48E/oP7iIDZawT+z81K8qxjBP2hARQSkI8U//Xs4YC/fxz+2oApn0SHYP/H3kQ9N0eA/5VLYp3Pt1j9oE/4vz4vRP4h1+N3UZuA///5bfao94T8=",
          "dtype": "f8"
         },
         "y": {
          "bdata": "KETAXDSiuL9ghNsEi2Grv7nCol2GB8C/Xg0ANbVItb/JlaKtY6nZPyiiKKa+nLY/yy6u8jlU3D+3pEYDHgPfPw7uNtmNn8A/e0wkGqvm3D9SqRMhMl2+P8jy3nUgZdw/DoniFn7h3T9gyAw5KA+/PwCGC1iBQ4g/AN4byT4nlL8ovMXWuoWiv3/Y8rt8WdW/2nhyPNMryb8rrMCd2WXNv7SHghALd7C/sHQjUDakr7/Bpg24fJnZP840pXLFdLk/7tUNNoqe3D/3D00p707Kv9fAAvU5oda/YK9cyScGw7/BlXOR5bHaP2nJ3QUU+d4/LopHsM9m3z8u99gYh3LfP0BfdThxRcE/ivgmlCGu2z8mFJlmMiy5P0APKZy9doM/HyfduKay2T9fFZMB5XndP7QoqEE0374/QPiIYSv+iz8=",
          "dtype": "f8"
         },
         "z": {
          "bdata": "KETAXDSimL8HJQvvM0+xv2A3bzmk5JE/+I27BpBkhr9WIsrsMOa5P67QS3XFl7E/bJx0u/Yaxz8it3MiL8vCP3CcrLz1N3y/Ts+2ZpT1uT+e1oVQH2+3PwPydfMyq8w/Rqef25jOxj/gytQy2gxtvwAIOZj0uke/gAM0+yCbmL/gy2cwTqyVv8bRIbgE4KA/WGa0ft6muz9ga5AeKJWzP6/VZSCxDq+/MOeMenfDk7/gOyBS2GLBP5hsUdMyGY2/pmXFVFKMvD/i58RcuQawP7sPpl5wMM0/CAXPXV9qjr8aVPVPedqzPzI32F/Bubg/q1jbLOmkuD9U3mHEDUu4P1hK4G8j3oy/2qdd4EvIuD/mW9Ve3hqzPwBSs/ydel8/8ojluq/yyz89hziujVPIP2RDjnaPDrk/gCWJW711fT8=",
          "dtype": "f8"
         }
        }
       ],
       "layout": {
        "height": 500,
        "margin": {
         "b": 0,
         "l": 0,
         "r": 0,
         "t": 30
        },
        "scene": {
         "aspectmode": "cube",
         "xaxis": {
          "title": {
           "text": ""
          }
         },
         "yaxis": {
          "title": {
           "text": ""
          }
         },
         "zaxis": {
          "title": {
           "text": ""
          }
         }
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "fillpattern": {
             "fillmode": "overlay",
             "size": 10,
             "solidity": 0.2
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermap": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermap"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Belief Trajectories: Prior Stays on Manifold, Posterior Moves Off"
        }
       }
      },
      "image/png": "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3XLtt2S8xksGFVwRBzmpbqe7Nche8V4SIAESIAESIAESIAF3BF7OkTh2OMUxdxvEq0InQHEsdKTskARIgARIgARIIB8JQAiDg2zZAzcPTu+pZ1+Vi6+6bTBuEfGL37j6dt/iGF7qw+UUpE3SeluIfPzqlz4n++29h2zWl9HL/vy8tLa1C148o51zwVVy6onHyrf+5UwrNg11gnozqHKPP/pr6axMSPX+41JOD8IEXvKbF+gHlk2TtS+uku3bNssxx59kRbmF1ezuK8w9Xa0zMyaEqarSmCXghSkyuFmTma+X2EREZOYqCtCN6GhftxFCYkW5iYU08012DjrtDa43tahyEWMJd2elnkW3gqJxvXUnelXUC1cwRcTp8xq1GC8ulmlHzJQPej+wsBXopqKOGGoY2tuge+y9eqnrq5MTPnNa2kfAXo8MAt/F373ZEuWvufwC6z58L/1EBftVr78t37noXJl5yAEyUR1fmxo63Txajtfg2anTen5sJEACJEACJEACJEAC2SPwyoe5cY4dthedY9nbZY405L+FtZ5DgN9vJkwSIAESIAESIAESGBkETIzihfPnyoLzzrAmvUFri52sQtSTKpihnljQBudNk4o9fhteNF/+/R/LBfPOkkMOGipGwbGBGDMIA39e8Ve59pZFsvQ3t0lBUeYXyHiB/vQffy/jTpjiODU4cjrUBVPXPz6yumIYHNGDZSp0pYt/dJqkEVHSuXX8sne6z8RVJsf8uRnHiCI4F9lyZblxiznNPRcuMjPfBo2oTGgtNC/NxFhib7LNN527Ld1ZgustivmuXvmSIDb1gOkHS80BE2VLYr1gP9s6h34f2Qlve9p97UD09fY778n5375JfvWz62X8uLrBZd77m8fVPfa83P/T66wxIcptbx4qynnZV1zLemNeifF6EiABEiABEiABEghO4NV1LcE78dHDoVMqfdzFW0ggOAE6x4IzZA8kQAIkQAIkQAJ5SgCC2MJ7HpM5px5n1RVbuHiJ3LXoUSte8YgZB1oxisufeXmImyzIUrrUGRLkpTAiE//68t/koV/cMOxlv1UXTIUlE3M2+0uXWE6NWZ+YmXHKv178U6n8+HgprnWOKTtk0nTpXt8hz67430giFO3umRatg5TJ6ea0KBO/58XFlRFQigu8uq/SzRcuo6hdWSNxvnAGoVZfGOchpqIMhOmo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I5YAxbERu3WcOAmQAAmQAAmQAAmERwDOsa998RTrj5eGmk3tXdGLA25j+PIhejATP8T54U8mISas2L9M8wnyeUm8QN19cWnViMV0MYuxwjFSW1lsXRMkjjHIXDPdi8jIGhWgMs0x2yJfWXGRziuWafqDn7+4cq1cfPXtMvezx1sCOGoKspEACZAACZAACZAACZAACZAACQwlQHGMJ4IESIAESIAESIAERjkBuEyOPn2BLL7tSl9uk2aNlIM4EnUz9cMaWro00rF/yHD2qD84eXIdPZiJRSaxCA4zL66sTONF+XmmaEIjBqbatyjn5advE5XoFPtohLFMApqfsVPdU6EialW5e2HM9AFn6NU33m05RH945dcdn2s8+/izu9YeZCMBEiABEiABEiABEiABEiCB0USA4tho2m2ulQRIgARIgARIgARSEHhp1Zty0VW3yfOPL7Repi+85zGpUrfJ5Rd9xTUvCFItWksr6gZ3D+LlEOlo4u8gmkFEaO/qzVtXUioudlGpeQc7I74grs/8LGqmYfRvogkR/2evi+bW8RfGHMLsw7j27I4445Kzn70wx0zuq1IFUvAL0lBf8M5Fjw5zkUEQW7h4idz74FKr+6NmTpPrrjhfJlMkC4Kb95IACZAACZAACZAACZAACYwgAhTHRtBmcaokQAIkQAIkQAIk4IYAaorhpTcErgvnz83oBrvpzl/J2nfWyRyNYbtLX6QfqS/KUXvM64vybAlkxnXVoXGOEJPiMfd1r9zwy+Y1EJWqVQQp1NhBiHsjUeSz87JHQZapaNmvBj+ISfh7pDW7GxFORazNbX21oGsNQxgzczAuMtHDtujWy60fw1X2xxWvWK6yI7U22VX67y1tHYOfB50/7ycBEiABEiABEiABEiABEiCBfCdAcSzfd4jzIwESIAESIAESIAEPBG6669ey/JmXrdphELwee2KFJZCh9pBTm3/pjbJGrzVi2hmzZ3kYceil2RLISlQQq6uKS1eizxIsRnobW1UscF1ta+oadMSN1DWZvUn09MlWXc9Ib+OriyWmjkWIfB0qYEbdwhTG7HOFWwz1x+Amu+GOXw6JUUWdsnn6PfD6n++JennsnwRIgARIgARIgARIgARIgATyggDFsbzYBk6CBEiABEiABEiABMIhgNphcIOcOOtQq0NEp8ENtuyBmx2dYBDUmlvafLnF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"
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = create_simplex_figure()\n",
    "\n",
    "# Prior trajectory\n",
    "prior_3d = np.array([belief_to_3d(b) for b in prior])\n",
    "fig.add_trace(go.Scatter3d(\n",
    "    x=prior_3d[:,0], y=prior_3d[:,1], z=prior_3d[:,2],\n",
    "    mode='lines+markers', marker=dict(size=3, color='darkgreen'),\n",
    "    line=dict(color='green', width=4), name='Prior (no obs)'\n",
    "))\n",
    "\n",
    "# Posterior trajectory\n",
    "post_3d = np.array([belief_to_3d(b) for b in posterior])\n",
    "fig.add_trace(go.Scatter3d(\n",
    "    x=post_3d[:,0], y=post_3d[:,1], z=post_3d[:,2],\n",
    "    mode='lines+markers', \n",
    "    marker=dict(size=4, color=np.arange(n_steps), colorscale='Blues', showscale=True,\n",
    "                colorbar=dict(title='Time', x=1.0, len=0.5)),\n",
    "    line=dict(color='blue', width=2), name='Posterior (with obs)'\n",
    "))\n",
    "\n",
    "fig.update_layout(title='Belief Trajectories: Prior Stays on Manifold, Posterior Moves Off')\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "line": {
          "color": "green",
          "width": 2
         },
         "name": "Prior",
         "type": "scatter",
         "y": {
          "bdata": "AAAAAAAAAABTW9o6WEyZPC4hCRSOmJM8U1vaOlhMmTzq+NKpfyqlPOr40ql/KqU8U1vaOlhMqTz87mNpM9WqPMAKHwDGSKw826+sB7O7sDzACh8AxkisPGJWY32dsrM8hR4VuY5UsjyFHhW5jlSyPECwGsSpv7c8S/tBPI1guTxZBvckXKi4PLm6RsorfsE8DLg29oE/wDw42LwKGwfCPDR4M7kXjMI8daY8b1MYxTxEOp2v0AXGPDiYEl/QL8g8QLAaxKm/xzzufurzuHnJPAAAAAAAAMU8vkwy3iKJyDw4mBJf0C/IPPiR5WYxp8w8vSAqxEvOzjyIfGZICRbNPP2Jw5Uvac08a/laMra/yzwB9uxUCMTNPEz5yFTA2s48zMoBVhVZzjxP2VyD+wvQPPWjq4QtetA8smIKhvqe0Tw=",
          "dtype": "f8"
         }
        },
        {
         "line": {
          "color": "blue",
          "width": 2
         },
         "name": "Posterior",
         "type": "scatter",
         "y": {
          "bdata": "chzHcRzHkT+qeqshuQOkP+CyUaPwLaQ/EEEtnzy5jT8w7bI+7QOkP3T9huRloIE/yDF4SLTdkj8glLDP32qQP99xUkgrlKY/VNLnkxoCmD+niFKQVVWDP0quhS8XGpE/bEAoa5LtmD/xlg9g3HCmP0wen61cx4M/7Oa+N6AZdD9Ip/M8qIV9Pxx8pzQOYZE/Zue42TcDmD+8GkDw1cWrP8YSenNWdq0/gKiqwMcqhT/Pgc9AM12iP2opHTJplKc/w32+eF5VmD8siipHu++WP4oFBMz0g4Q/vEbhIg/wkj+fmpToIFCaP886oTwmOpI/sdR3VibMkD+Ug4BzF6eQP8tVvDY5Xac/lzmR/f4Foz84ClvvbXaCPzbDB7VJ33c/0izDdQBRmz/sMaxTtmiZPydY2adSSIA/fkdahpO7cz8=",
          "dtype": "f8"
         }
        }
       ],
       "layout": {
        "height": 300,
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "fillpattern": {
             "fillmode": "overlay",
             "size": 10,
             "solidity": 0.2
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermap": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermap"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Coupling: Distance from Independence"
        },
        "xaxis": {
         "title": {
          "text": "Time"
         }
        },
        "yaxis": {
         "title": {
          "text": "||P(S₁,S₂) - P(S₁)P(S₂)||"
         }
        }
       }
      },
      "image/png": "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NTnCOGQ6j46CABOQqLJz5y8pFuJEauOIAhgIWI2AOMddvHSZLv8Xrdb+nkj6lP3EvpHVRh39SU3A832fREm87sl2wzsw9o0wZ0AABIxMQFNxzMgdRdtAAARAAARAAARAAARAAARAAARAAARAAARAAARAAARAAARAAARAAOIY5gAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgEDIEIA4FjJDjY6CAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAhAHMMcAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQCBkCmopjfYa9Tzv3feETvGJFH6WJgzv7dA4OBgEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAF/CGgqjh347Hs6dfa8T+3IkT0Llfjfkz6dg4NBAARAAARAAARAAARAAARAAARAAARAAARAAARAAARAAARAAARAwB8Cmopj/jQA54AACIAACIAACIAACIAACIAACIAACIAACIAACIAACIAACIAACIBAoAhAHAsUadwHBEAABEAABEAABEAABEAABEAABEAABEAABEAABEAABEAABEAg6AQgjgV9CNAAEAABEAABEAABEAABEAABEAABEAABEAABEAABEAABEAABEACBQBGAOBYo0rgPCIAACIAACIAACIAACIAACIAACIAACIAACIAACIAACIAACIBA0AlAHAv6EKABIAACIAACIAACIAACIAACIAACIAACIAACIAACIAACIAACIAACgSIAcSxQpHEfEAABEAABEAABEAABEAABEAABEAABEAABEAABEAABEAABEACBoBOAOBb0IUADQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEAkUA4ligSOM+IAACIAACIAACIAACIAACIAACIAACIAACIAACIAACIAACIAACQScAcSzoQ4AGgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIBIoAxLFAkcZ9QAAEQAAEQAAEQAAEQAAENCNw6Msf6fJ/V255PbvdTiWLFaatez6lOzJloGJFH9Ps3rgQCIAACIAACIAACIAACIAACICAuQlAHDP3+KH1IAACIAACIAACIAACIBCSBGq1fouOHDtx275/v2cOlanXjR554F6aMqxrSHJCp0EABEAABEAABEAABEAABEAABG4kAHEMswIEQAAEQAAEQAAEQAAEQMB0BFwJCZSU5G72z7+eoLptBtCkIZ2pxPOF1f/ZbEQOjh67dPkK2cPCKGOGSNP1EQ0GARAAARAAARAAARAAARAAARDQhwDEMX244qogAAIg8P/27i3EqjIMA/CnoamZ2pidhBrSFDUVghgp7IAVkdrR0jE6aZYORJaTXZimTJQyotbFKDIQJtGQUaRFEHQUKumolVoqZIqmWSQmRmC23eEwUoT6qzP/nmdfLubb6/2fb+5e1l4ECBAgQIAAgZMksKFQjt183xNR98wjxZ9SbPqZNW9JnHd295hw54jY98efMaG6Nm4YNiQ+W70+Vq76Os49qyymTBwdZ5Z1jWfrX4kvv9lY+AnG/jG+cngM7t+r8as2b90Rcxc2xCdfrIsOp7aLoRWDonrSmCjrdvpJOqXbECBAgAABAgQIECBAgMDxElCOHS9J30OAAAECBAgQIECAQLMI/F85VllVE73Le0bN1HGxp/AU2ZARVcWMI6+7rFh+LX/7o1izdlPx2qgRV0bfXufHshXvxf79f8XyJU8Xr+/c9VtcPWpyXDKwT9wx8qr4dfeeqH/xjRjQtzwWzZnSLGd2UwIECBAgQIAAAQIECBA4dgHl2LHbmSRAgAABAgQIECBAoAUIHG05Nu3hu2LsLcOKyVcXirGxhQKtdvqkwhNlFcVrK1etiYmPz4t3ls2Lc3qURW1dQ7y84v344NUF0anjPz/P2PD6u1Ez/4X48LXnovsZXVqAgggECBAgQIAAAQIECBAgcKQCyrEjlfJ3BAgQIECAAAECBAi0SIGjLceaFmFbtu2M68dOLT4BNrRiYPF8a7//IW5/YGa8VDc9BhWeLrt38uz49Kv10e+iCxrPf/AptK3bf45li2dG/z7lLdJFKAIECBAgQIAAAQIECBD4bwHlmP8MAgQIECBAgAABAgSyFkgpx7b9tCuuHVN9WDm2fuOPcdv9MxrLsdEPzoq2p7SNqntu+pfT4AG9o0vnTln7CU+AAAECBAgQIECAAIHWJqAca20bd14CBAgQIECAAAECJSZwosuxabPr4+PPv403l86Jjh3aN+odOHAg2rRpU2KajkOAAAECBAgQIECAAIHSF1COlf6OnZAAAQIECBAgQIBASQuc6HJs3YbNMWrCk3HFkMEx8e4bo/NpHePg02XPN7wV9XMfi25dO5e0r8MRIECAAAECBAgQIECg1ASUY6W2UechQIAAAQIECBAg0MoEDpVji+Y8Wnhv2KDDTl9ZVRO9y3tGzdRx8fvefVExfFI0fefY9h2/xDWjp8Ti2uq4/NKLi7PfbdoSt46fHg0LZ8TAfhcWr61ctSaeWrC0+J6xQ5+D7yibP+uhw54ma2X0jkuAAAECBAgQIECAAIEsBZRjWa5NaAIECBAgQIAAAQIEmkNg9569xZKtR1nXaN++XXNEcE8CBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhHQDmWz64kJUCAAAECBAgQIECAAAECBAgQIECAAAECBAgQSBRQjiUCGidAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMhH4G8la2yHVUZ6awAAAABJRU5ErkJggg=="
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Prior coupling:  mean=0.000000 (should be ≈0)\n",
      "Posterior coupling: mean=0.0233\n"
     ]
    }
   ],
   "source": [
    "# Coupling over time\n",
    "fig = go.Figure()\n",
    "fig.add_trace(go.Scatter(y=prior_coupling, name='Prior', line=dict(color='green', width=2)))\n",
    "fig.add_trace(go.Scatter(y=posterior_coupling, name='Posterior', line=dict(color='blue', width=2)))\n",
    "fig.update_layout(\n",
    "    title='Coupling: Distance from Independence',\n",
    "    xaxis_title='Time', yaxis_title='||P(S₁,S₂) - P(S₁)P(S₂)||',\n",
    "    height=300\n",
    ")\n",
    "fig.show()\n",
    "\n",
    "print(f\"Prior coupling:  mean={prior_coupling.mean():.6f} (should be ≈0)\")\n",
    "print(f\"Posterior coupling: mean={posterior_coupling.mean():.4f}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}