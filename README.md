# lewm-atari
This is a minimal adaptation of LeWorldModel to online setting in the Atari environment. The dynamics model is learnt in spirit with LeWorldModel: a CNN encoder encodes observations into embeddings and a gru based predictor predicts the embedding of the current observation conditioned on previous observations and actions. The encoder and predictor are trained using prediction and sigreg losses. The behaviour model is learnt using the Dreamer-v3 famerwork: a reward model and a discount model are learnt as probes over the learnt dynamics model in a detached state. Actor and Critic models are trained using imagined rollouts over the learnt dynamics model.

<table>
  <tr>
    <td>
      <img src="assets/episodes.gif" width="360" height="2160" alt="Episodes GIF">
    </td>
    <td>
      <img src="assets/plots.png" width="360" height="2160" alt="Training Plots">
    </td>
  </tr>
</table>

### installation
```
git clone lewm-atari
cd lewm-atari
pip install requirements.txt
```

### run
```
cd scripts
python lewm_atari.py
```

You can train on any atari game by replacing the game hyperparameter in lewm_atari.py with a game listed [here](https://ale.farama.org/environments/complete_list/)
