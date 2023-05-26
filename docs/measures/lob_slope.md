---
tags:
  - Market Microstructure
  - Liquidity
  - Limit Order Book
---

# Limit Order Book Slope

## Introduction

Some various slope measures of the limit order book (LOB). A steeply sloping order book indicates a lack of depth beyond the best bid or ask quote, a sign of illiquidity.

!!! warning "Note"
    The notations below largely follow the respective papers. Therefore, the same symbol may have different meanings.

    Certain measures are estimated by (typically hourly) snapshots or intervals, where quantities supplied at each level are aggregated. If so, the functions here should be applied on data by intervals.

## Næs and Skjeltorp (2006)

[Næs and Skjeltorp (2006)](https://doi.org/10.1016/j.finmar.2006.04.001) introduce the slope measure of bid, ask and the LOB. They measure the slope at each price level by the ratio of the increase in depth over the increase in price, then average across all price levels.

$$
\begin{equation}
  \text{Bid Slope}_{it} = \frac{1}{N^B} \left\{\frac{v^B_1}{|p^B_1/p_0 -1|} + \sum_{\tau=1}^{N^B-1} \frac{v^B_{\tau+1}/v^B_{\tau}-1}{|p^B_{\tau+1}/p^B_{\tau}-1|} \right\}
\end{equation}
$$

$$
\begin{equation}
  \text{Ask Slope}_{it} = \frac{1}{N^A} \left\{\frac{v^A_1}{p^A_1/p_0 -1} + \sum_{\tau=1}^{N^A-1} \frac{v^A_{\tau+1}/v^A_{\tau}-1}{p^A_{\tau+1}/p^A_{\tau}-1} \right\}
\end{equation}
$$

where $N^B$ and $N^A$ are the total number of bid and ask prices (tick levels) of stock $i$,[^1] and $\tau$ is the tick level. The best bid (ask) price is denoted by $p_1^B$ ($p_1^A$), i.e. $\tau=1$. $p_0$ is the bid-ask midpoint. $v_{\tau}^B$ and $v_{\tau}^A$ denote the natural logarithms of the _accumulated_ volume at each tick level $\tau$ on the bid and ask sides.

[^1]: For example, if we examine 5 levels of LOB, $N^B$ and $N^A$ will be 5.

!!! note
    If we define $V^B_\tau$ as the total volume demanded at $p^B_\tau$, then $v_{\tau}^B = \ln \left(\sum_{j=1}^\tau V_j^B\right)$.

The LOB slope for stock $i$ at time $t$ is then computed as

$$
\begin{equation}
  \text{LOB Slope}_{it} = \frac{\text{Bid Slope}_{it} + \text{Ask Slope}_{it}}{2}
\end{equation}
$$

Intuitively, the bid (ask) slope measures the percentage change in the bid (ask) volume relative to the percentage change in the corresponding bid (ask) price, which is averaged across all limit price levels in the bid (ask) order book, and the LOB slope is an equally weighted average of the bid and ask slopes.

## Ghysels and Nguyen (2019)

[Ghysels and Nguyen (2019)](https://doi.org/10.3390/jrfm12040164) take a more direct approach. The slope is estimated from regressing the percentage of cumulative depth on the price distance at each level.

$$
\begin{equation}
QP_{it\tau} = \alpha_i + \beta_i \times |P_{it\tau} - m_{it}|  + \varepsilon_{it\tau}
\end{equation}
$$

where $QP_{it\tau}$ is the percentage of cumulative depth at level $\tau$ at time $t$ and $|P_{i\tau} - m_{it}|$ is the distance between the price at level $\tau$ and the best bid-ask midpoint. $\beta_i$ is the estimated slope. The bid and ask slope are estimated separately.

Because the cumulative depths are standardized by the total depth across all $K$ levels, $QP_{itK}$ is always 100%. As such, the slope measure $\beta$ depends only on how wide the $K$-level pricing grid is (the intercept $\alpha$ takes care of $QP_{it1}$). Accordingly, this slope measure captures the tightness of prices in the book. A steeper slope indicates that limit orders are priced closer to the market. Another way of interpreting the slope is that it measures the elasticity of depth with respect to price. A steeper slope implies that for one unit increase in price, there is a greater increase in the quantity bid or offered, implying a greater willingness of liquidity providers to facilitate trading demand on each side of the market.

## Hasbrouck and Seppi (2001)

Let $B_k$ and $A_k$ denote the per share bid and ask for quote record $k$, and let $N_k^B$ and $N_k^A$ denote the respective number of shares posted at these quotes. Thus, a prospective purchaser knows that, if hers is the first market buy order to arrive, she can buy at least $N_k^A$ shares at the ask price $A_k$.

!!! note
    These slope measures below are for the whole LOB, not specific to bid or ask.

### Quote Slope

$$
\begin{equation}
  \text{Quote Slope}_k = \frac{A_k - B_k}{\log N^A_k + \log N^B_k}
\end{equation}
$$

### Log Quote Slope

$$
\begin{equation}
  \text{Log Quote Slope}_k = \frac{\log (A_k / B_k)}{\log N^A_k + \log N^B_k}
\end{equation}
$$

## Della Vedova, Gao, Grant and Westerholm (working paper)

Slope is measured by the cumulative volume at level $K$ divided by the distance from the $K$-th level price to the bid-ask midpoint.

$$
\begin{equation}
  \text{Bid Slope}_{it} = \frac{\sum_{x=1}^K \text{Bid Depth}_{itx}}{|\text{Bid Price}^K_{it} - m_{it}|}
\end{equation}
$$

$$
\begin{equation}
  \text{Ask Slope}_{it} = \frac{\sum_{x=1}^K \text{Ask Depth}_{itx}}{\text{Ask Price}^K_{it} - m_{it}}
\end{equation}
$$

where $\text{Depth}_{itx}$ is the sum of the quantity of available at bid/ask depth level $x$ in stock $i$ at time $t$, $\text{Price}_{it}^K$ is the bid/ask price at the $K$-th level, and $m_{it}$ is the bid-ask midpoint at time $t$.

### Scaled Depth Difference

In addition, the relative asymmetry of depth is also informative about the relative demand and supply of the stock. Scaled Depth Difference (SDD) is constructed to capture the relative asymmetry in the LOB at a particular time at a particular level $x$.

$$
\begin{equation}
 SDD_{itx} = \frac{\text{Ask}_{itx} - \text{Bid}_{itx}}{\text{Ask}_{itx}+\text{Bid}_{itx}} 
\end{equation}
$$

where $\text{Ask}_{itx}$ is the ask price of stock $i$ at level $x$ at time $t$ and $\text{Bid}_{itx}$ is the bid price of stock $i$ at level $x$. SDD represents a scaled level of asymmetry at the prevailing quote to level $x$, ranging from -1 and 1. A value of SDD greater than zero indicates asymmetry in the direction of the ask side of the book.

## API

### ::: frds.measures.lob_slope

## TODO

- [ ] Implementation of all other slope measures.

## References

* [Næs and Skjeltorp (2006)](https://doi.org/10.1016/j.finmar.2006.04.001.), Order Book Characteristics and the Volume–Volatility Relation: Empirical Evidence from a Limit Order Market, _Journal of Financial Markets_ 9(4), 408–32.
* [Valenzuela, Zer, Fryzlewicz and Rheinländer (2015)](https://doi.org/10.1016/j.finmar.2015.03.001), Relative liquidity and future volatility, _Journal of Financial Markets_, 24, 25–48.
* [Ghysels and Nguyen (2019)](https://doi.org/10.3390/jrfm12040164), Price discovery of a speculative asset: Evidence from a bitcoin exchange, _Journal of Risk and Financial Management_, 12(4), 164.
* [Duong and Kalev (2007)](http://dx.doi.org/10.2139/ssrn.1009549), Order Book Slope and Price Volatility, _SSRN_.

## See Also

---

[:octicons-bug-24: Bug report](https://github.com/mgao6767/frds/issues/new?assignees=mgao6767&labels=&template=bug_report.md&title=%5BBUG%5D) | [:octicons-heart-24: Sponsor me](https://github.com/sponsors/mgao6767)