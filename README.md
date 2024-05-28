# A faster algorithm for brain change estimation

The estimation brain atrophy can be crucial in the
evaluation of brain diseases and can help to analyze their progression. Existing methods compute the atrophy map from two
Magnetic Resonance Imaging (MRI) scans of the same subject,
which has limitations in terms of evaluation time, often due to
the multi-step process. In this work we proposed a new technique
for atrophy map calculation. It is designed to estimate the change
between two MRI scans with the aim of significantly reducing the
execution time. The consecutive subject time points are evaluated
by a simple U-net which shows the goodness of a single-step
process. We train and evaluate our system on a dataset consisting
of 2000 T1-weighted MRI scans sourced from ADNI and OASIS dataset.
Experimental results demonstrated a considerably reduction in
execution time while maintaining the performance in terms of
atrophy mapping. We believe that this pipeline could significantly
benefit clinical applications regarding the measurement of brain
atrophy.

