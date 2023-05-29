import numpy as np
import matplotlib.pyplot as plt # to plot graphs
import nengo

from nengo.utils.ensemble import sorted_neurons # for sorting neurons
from nengo.utils.matplotlib import rasterplot # for raster plots for neurons in an ensemble

model =  nengo.Network(label = "Many Neurons") # create a network
with model: # add ensembles to the network
    A = nengo.Ensemble(100, dimensions=1) # 100 neurons in one-dimensional signal

with model:

    # input node that provides a sine wave input
    # lambda is used to define a function without a name
    # t is the time variable that is passed to the function
    # np.sin(8 * t) is the function that is defined
    sin_probe = nengo.Node(lambda t: np.sin(8 * t))  # Input is a sine

    nengo.Connection(sin_probe, A, synapse = 0.01) # connect the input to the ensemble with 10ms filter

    # A_probes are the output of the ensemble
    # A_spikes are the spikes of the neurons in the ensemble
    # Spikes are the output of the neurons
    # Probes are the output of the ensemble
    # Difference between ensemble and neurons are that neurons are the individual neurons in the ensemble and ensemble is the group of neurons

    sinProbe = nengo.Probe(sin_probe) # probe the input
    A_probe = nengo.Probe(A, synapse = 0.01) # probe the output
    A_spikes = nengo.Probe(A.neurons) # probe the spikes

with nengo.Simulator(model) as sim:
    sim.run(1.0) # run the model for 1 second
