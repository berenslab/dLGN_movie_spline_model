# Spline-based generalized linear model of dLGN neuron activity

## Project Description

This repository contains code for analyzing the modulation of neuronal activity in the dorsolateral geniculate nucleus (dLGN) of the thalamus in awake mice. 
The project aims to investigate how stimulus-driven signals are combined with modulatory inputs such as corticothalamic feedback and behavioral state.

We utilized the [RFEst Python toolbox](https://github.com/berenslab/RFEst) for implementing a spline-based generalized linear model (GLM) to estimate spatio-temporal receptive fields (STRFs) of dLGN neurons.
We extended the model to integrate additional spline kernels for pupil size, locomotion, and corticothalamic feedback.

## Installation

1. Clone this repository:
    https://github.com/berenslab/dLGN_movie_spline_model.git
2. Create a virtual environment and activate it:
    python -m venv venv
    source venv/bin/activate
3. Install the required packages: 
    pip install -r requirements.txt
    
## License
This project is licensed under the CC0-1.0 License.
