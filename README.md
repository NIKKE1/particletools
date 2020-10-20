# particletools


## Event inspector:

run: python3 event_inspector.py .

Calculates delta_R for particles based on particle id, mass, and process cuts. 
Calculates density of selected particles, and plots them on a Particle id/mass -axis.
Uses multiprocessing.

Currently calculates and plots the delta_R for W-bosons, and plots its mass range around 70-120GeV

## fj_w and gen_w

For each event the script selects W-bosons based on pgid:24, mass>0, isHardProcess : 7.
Calculates delta R to each FatJet and selects closest FatJet in the event based on smallest delta R.
Plots masses and deltaR's of these FatJets

Run with:
python3 fj_w.py
python3 gen_w.py
