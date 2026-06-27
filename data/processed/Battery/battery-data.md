# Occurances in modelling phase
table3_diagnostic_cycles is thr 3rd table on Page 6 of https://pangea.stanford.edu/ERE/pdf/OnoriPDF/Journals/72.pdf paper, the source of our data regarding Ion-Lithium Lifecycle and Aging. 

Using that table, we can cross-reference certain points of interest with our data itself, and subsequently with the DEVST EV Data. 

The first step in this pipeline is to go through ```fase1_battery_processing.py```. 

## The problem 
Table 3 originally is a partial snapshot of the 1st of February, 2022, but the campaing went on for 23 whole months. The real dataset continues accumulating diagnostics after that date. Cells as G1, V4, W10 and others have 6 ```diag_number``` extras in the real CSV that do not exist in the paper, yet. 

## The Solution
First, we thought of extrapolating the missing data as 25 cycles, but that would be incorrect, as 25 isn't a fixed cycle count. Instead, we extrapolated using the last known interval of each cell (not global constant), and marking the estimated values.

As they were stable, around 25-29 cycles, main() was adjusted with the interpolation method.

Other Observations include:
W4 and W5 showed small non-monotonic segments in the actual SoH (capacity increased slightly between two consecutive diagnostics)—this is measurement/relaxation noise typical of capacity tests, not a calculation error. It is worth considering the use of smoothing (e.g., moving average) if this affects the curve fit later on.

The initial capacities (diag_number=1) were all within ±1.1% of the factory-rated Qnom (4.85 Ah)—a great sign that the Coulomb count is correct.

G1 shows a high proportion of extrapolation (6 out of 11 diagnostics are estimated)—treat the predictions in this cell with high EFC with greater caution.

