# Reinforcement Learning for Dynamic Supply Chain Congestion Control

Website: https://arihanth26.github.io/sc-rl-congestion/

## Overview

This project applies Reinforcement Learning (RL) to a realistic supply chain control problem: routing demand across multiple fulfillment centers under capacity constraints and demand volatility.

In real fulfillment networks, demand arrives in bursts. Greedy or static routing policies often perform well on average but fail during peak periods, causing queue buildup, late orders, and cascading congestion. This project models fulfillment routing as a sequential decision-making problem and evaluates how RL-based policies can improve stability, service levels, and long-term cost.

The system is driven by real-world demand patterns derived from large-scale public data and is built as a modular simulator suitable for baseline comparison and RL experimentation.

## Problem Statement

Given:

Time-varying, region-level customer demand

Multiple fulfillment centers with limited processing capacity

Costs associated with shipping, backlog, and congestion

We aim to learn a routing policy that:

Avoids congestion collapse during demand spikes

Balances load across fulfillment centers

Minimizes long-term operational cost and service penalties

This is a control problem under uncertainty, not a forecasting problem.

## Key Features

Discrete-time fulfillment network simulator with queue dynamics

Real, high-frequency demand data (non-synthetic)

Multiple baseline routing policies for comparison

Explicit modeling of delayed congestion effects

Modular design that supports RL training and extensions
