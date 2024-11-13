# Minesweeper implementation & custom AI player support

This faithful implementation of Minesweeper can either be played as usual by a player or by a custom AI agent that interfaces with the API. It supports both a graphical mode (so you can watch the AI play -- helpful for debugging), and a non-graphical mode that is optimised for performance.
In addition, the view of the board that is fed to the AI agent can be restricted to a NxM rectangular sample (of custom dimension) which is scanned across the entire board.

Built with Python.


## Why does this exist?
The motivation behind this version of Minesweeper was to provide an efficient implementation of minesweeper that can easily be interacted with a custom AI implementation.

It was used as the basis for a dissertation (during undergraduate studies) in which two custom AI agents were design, implemented, and their performances compared. The two agents are a perfect-solver AI (using a CSP-solver with random guesses when no definite solutions are possible) and a pattern-matching heuristic player which tries to closely mimic the strategy a human player would use for deducing which tiles to flag/click (without resorting to brute-force calculations).

In particular, the motivation was to analyse how restricting the information available to an AI agent could upper-bound its best possible performance (as measured by the perfect-solver) in terms of win rate, so as to find the minimum information that should be available to any generic AI agent for it to have any hope of still having a significant win-rate. The pattern-matching strategy was analysed to see how a more typical strategy fares in the same restricted circumstances.
The two built-in AI players are:
- An exact solver -- always plays a definite move if its possible to deduce it using all its available information. Otherwise, make a uniformly random guess. This was done by transforming the (NxM sample view of) the board into a constraint satisfaction problem and then using a CSP solver on that.
- A naive solver -- mimics the typical strategy a person would use by analysing the patterns of numbers along the borders

The information restrictions were:
- the NxM sample of the grid that the agent can see at any one time (which is then scanned across the board)
- whether an agent can flag tiles

For a more detailed discussion and analysis of the results, see the full report in the repository.

![](minesweeper-demo.png)
