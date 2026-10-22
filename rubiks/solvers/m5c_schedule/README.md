# Method 5C: net-gain scheduling of bulk commutators

**Owner:** Gaurav · **Registered name:** `m5c_schedule` · **Builds on:** Method 3 (technical basis, Section 6.4)

Demaine's algorithm batches clusters that need the *same* fix, and Method 3's 3-cycle version
(Section 6.4) applies one commutator type over a block of layers X × Y only when every cell of
the block helps its cluster. Once the clusters diverge those blocks shrink to single cells, and
batching fades: on the 20×20×20, 358 of Method 3's 385 center rounds fixed a single orbit.

5C changes only the schedule:

1. **Net-gain blocks.** A block is worth applying when its *net* slots fixed per move beat the
   best single-cluster 3-cycle, cells that do not help (or slightly hurt) included:
   maximise Σ gain(i, j) over X × Y minus λ · moves, with λ the best single-cluster
   slots-per-move right now. X and Y by alternating maximisation; every candidate is applied to
   the state and scored exactly (cells hitting one cluster compose; nothing outside the
   clusters may move).
2. **Setup sharing.** A commutator conjugated by the same outer setup as the previous one is
   charged 2·|setup| less; the shared `s' s` cancels when moves are canonicalised.
3. **Lookahead** (one step, N ≤ 15 by default): the best four candidates are re-scored by the
   best step available after them.

Square: the interior clusters' net-gain schedule (gain = drop in exact distance) and Method 3's
schedule are both computed (milliseconds) and the shorter is kept, so 5C is never longer than
Method 3 there. Every other stage (wing parity, 3×3×3 stage, Square frame and edges) is Method
3's, so 5C vs 3 isolates the schedule.

Results: `docs/STATUS.md`. Tests: `tests/test_m5c.py`.
