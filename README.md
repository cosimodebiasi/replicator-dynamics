# replicator-dynamics
This is a Python implementation of a quasi-replicator dynamics model with innovation, market selection, and firm entry-exit.

The simulation tracks:

- firm-level productivity;
- firm-level market shares;
- productivity deviations from the weighted market average;
- market turbulence;
- market concentration through the Herfindahl index;
- firm size growth rates;
- size-rank relationship.

The script normalises market shares after the entry-exit step so that aggregate market share remains equal to one in each period.
