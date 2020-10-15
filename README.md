# particletools


## Event inspector:

run: python3 event_inspector.py .

Calculates delta_R for particles based on particle id, mass, and process cuts. 
Calculates density of selected particles, and plots them on a Particle id/mass -axis.
Uses multiprocessing.

Currently calculates and plots the delta_R for W-bosons, and plots its mass range around 70-120GeV
