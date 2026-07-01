# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GLOP is a unified hierarchical framework that efficiently scales toward large-scale routing problems. It partitions large routing problems into Travelling Salesman Problems (TSPs) and TSPs into Shortest Hamiltonian Path Problems (SHPPs). The authors hybridize non-autoregressive neural heuristics for coarse-grained problem partitions and autoregressive neural heuristics for fine-grained route constructions. See `README.md` for usage and additional documentation.

I am currently exploring more ideas on decomposition based on GLOP. Specifically, I am considering methods that would make the decomposition of TSPs into SHPPs more efficient. The current implementation (refer to `/docs/LOCAL_CONSTRUCTION.md`) is good but it requires many rounds of refinements.

## Dev

The current project uses the conda env `glop` with all dependencies for GLOP installed.

`pytest` and `ruff` are included, make good use them to create tested and properly formatted code.