# Minesweeper implementation with custom AI player support

An accurate replica of Minesweeper extended with an API allowing for custom AI player integration. It supports both:

- **Visual mode** – Observe AI play in real-time (useful for debugging and analysis), or play the game manually if preferred.
- **Headless mode** – Optimised for performance

Additionally, AI players can be restricted to viewing only a rectangular sample of the board (of configurable dimensions) scanned across the entire grid per turn. This feature enables testing of AI decision-making under limited information constraints.


Built with Python.

![](minesweeper-demo.png)

## Key Features
- **Two AI implementations included**:
  - *Perfect deterministic solver* – Uses brute-force constraint satisfaction (Google's CP-SAT Solver) to select a tile that is guaranteed to be safe whenever it is possible to know this in a given turn. When there are no guaranteed safe moves, it selects a random tile instead.
  - *Pattern-matching solver* – Mimics human-like strategy by identifying [number patterns along tile borders](https://minesweeper.online/help/patterns).
- **Visual / Headless Modes** – Play the game manually or watch an AI play using the visual mode. Headless mode allows for high-speed simulations without rendering allowing for the AI's performance to be more easily benchmarked
- **Restricted View Configurations**: AI players can be limited to an NxM rectangular view. Each turn, the AI is provided NxM samples scanned across the board.


## Motivation
This project was developed out of a need for an efficient Minesweeper implementation that also exposes an interface allowing custom AI players.

It was originally used in an undergraduate dissertation that explored the maximum attainable performance, as measured by win rate, of optimal deterministic players (using brute force search) under a variety of information constraints (e.g., sample size, whether flagging tiles is allowed). The goal was to determine the smallest amount of information required for an AI to achieve a significant win rate.

The pattern-matching solver was also analysed to see how the more typical strategy used by human players fares in the same restricted circumstances.

For a more detailed discussion and analysis of the results, see the full report in the repository.



