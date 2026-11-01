---
title: "Getting Spain's Data Center Decree Right: Hourly Matching That Cuts Emissions and Costs"

summary: Should new data centers in Spain match their power use hour by hour with new wind, solar and storage? Or does that kill investment? 

abstract: |
  The next decade of digital infrastructure will be built somewhere and powered by something. Spain is deciding how. A [draft royal decree](https://www.miteco.gob.es/es/energia/participacion/2026/detalle-participacion-publica-k-851.html) would require new data centers to back, in every hour, at least 80% of their consumption with renewable electricity generated that same hour by new Spanish installations, with forfeiture of grid access on non-compliance. The stakes are measured in gigawatts: data centers hold 6.3 GW of granted transmission access, and more than 12 GW of access and connection rights granted since 2021.

  The policy's logic: additional demand must bring additional clean supply, matched at the hours it actually consumes. Done right, this makes new load a system asset. It finances new wind, solar and storage, caps its own marginal emissions, and hedges the operator against wholesale price volatility. Without it, every new flat gigawatt is served by the gas margin: higher emissions, deeper gas import dependency, higher consumer prices, and data centers exposed to every price swing. Bad for Spanish society, and bad for the data center operators themselves.

  This work is done in the open and in the public interest.

  **Results.** The results show that dropping the decree's core rule, that new data centers match their consumption *hour by hour* with power from *new* renewable and storage assets, is a lose-lose outcome for Spain and for data centers. Without it, the demand of a new data center is met mostly by gas power plants. Take one example: a 500 MW data center in Aragon (a small part of the 6.3 GW of grid access data centers already hold) makes Spain burn 10% more gas for power, emit about 1 Mt more CO2 a year, and pay almost 300 M€ more a year to run its power system. Gas sets the price, so the spot price rises by about 1.3 €/MWh for every consumer, the data center included. With hourly matching in place at 80% or 90% of consumption, that rise and the added emissions shrink by roughly the same share, and the data center covers it with its own renewable and storage contracts, shielded from the gas-set spot price. How emissions are counted matters: the average grid mix suggests about 70 gCO2/kWh for the data center's consumption (attributional), while the actual change in emissions is four times larger (consequential), because gas serves the new demand.

  One clause of the draft decree needs fixing: it asks for 80% *in every hour*, not 80% *hour by hour* over the year, the usual meaning of hourly matching (as in [Google's CFE score](https://sustainability.google/reports/24x7-carbon-free-energy-methodologies-metrics/)). Meeting 80% in every hour means covering even the darkest, calmest winter hours, which takes a portfolio that overshoots the rest of the time: it reaches about 99% under the usual definition, costs data centers more than buying from the grid, and costs the system more than having no rule at all. With that one fix, the decree delivers most of the social benefits, and data centers pay less than they would for grid power at the price their own demand pushes up. It also cuts Spain's dependence on imported gas and the data centers' exposure to gas price swings.

  **Update, 27 October 2026.** The revised draft of 26 October adds an optional five-year transition for data centers that accept automatic curtailment: the hourly floor then applies only in peak tariff periods (60% in P1 to 2028, then 70% in P1 and 60% in P2). Both stages are modelled and included as scenario options. Hourly matching on an annual average cuts emissions more, at the same or lower cost, than these transition rules: 80% on average beats the 2026 rule on both counts, and 90% on average matches the 2029 rule on cost with half the emissions increase and less of the cost passed on to other consumers.

authors:
- Killian Daly
- Iegor Riepin

date: "2026-10-29T00:00:00Z"
publishDate: "2026-10-29"
featured: false

image:
  caption: 'Change in the Spanish power system with a 500 MW data center in Aragon under each procurement rule. Image credit: [Riepin (2026)](https://github.com/pypsalabs/spain-datacenter-decree)'
  focal_point: ""
  preview_only: false

publication: '*EnergyTag and PyPSA Labs*'
publication_short: ""
publication_types:
- "4"

doi:
url_pdf: https://energytag.org/wp-content/uploads/2026/10/20261029-FINAL-Getting-Spains-Data-Center-Decree-Right_-Hourly-Matching-That-Cuts-Emissions-and-Costs-v4-1.pdf
url_code: https://github.com/pypsalabs/spain-datacenter-decree
url_dataset:
url_preprint:
links:
- name: LinkedIn Post
  url: https://www.linkedin.com/feed/update/urn:li:activity:7521614766411517952/
  icon_pack: fab
  icon: linkedin
- name: Draft royal decree (MITECO)
  url: https://www.miteco.gob.es/es/energia/participacion/2026/detalle-participacion-publica-k-851.html
slides:
---
