# Benchmark dataset verification log

Status of each Regime A entry in `data_collection/experimental_benchmark_data.py`
checked against its cited source. First pass (2026-11-23) used abstracts and
index records only; no full texts were read. "Needs full text" means the value
must be confirmed from the paper's figure or table before it is used.

| Entry in file | Cited source | Finding | Status |
|---|---|---|---|
| Ti3C2Tx film, 45 um, 92 dB, sigma 4600 S/m | Shahzad et al., Science 2016 | 45 um and 92 dB confirmed. Conductivity is reported in S/cm (about 4500-4600 S/cm), so the file value is 100x too low. | Unit error; needs full text for exact sigma |
| Ti3C2Tx/SA film, 8 um, 57 dB, sigma 2500 S/m | Cao et al., Adv. Funct. Mater. 2020 | 57 dB for 90 wt% Ti3C2Tx-SA is from Shahzad et al. 2016, not Cao 2020. Thickness and sigma not confirmed; sigma likely S/cm. | Wrong citation; needs full text |
| Ti3C2Tx/cellulose nanofibre, 47 um, 72 dB, sigma 1800 S/m | Han et al., ACS Nano 2020 | Han 2020 is "Beyond Ti3C2Tx". The MXene/CNF nacre-like paper is Cao et al., ACS Nano 12(5), 2018. Values not confirmed. | Wrong citation; needs full text |
| MXene film 10 um at 28 GHz, 50 dB, sigma 4600 S/m | Wanasinghe et al., Nanomaterials 2020 (review) | Secondary source; same sigma unit problem. | Needs primary source |
| MWCNT/PMMA 0.5, 2, 5, 10 wt% | Arjmand et al., Carbon 2011 | Paper is MWCNT/polycarbonate, not PMMA. Values not confirmed. Sigma entries are round numbers (1e-4, 0.5, 10, 100 S/m). | Wrong material; needs full text |
| MWCNT/PVDF 7 wt%, 36 dB | Al-Saleh & Sundararaj, Carbon 2009 | Paper is MWCNT/polypropylene, not PVDF. Values not confirmed. | Wrong material; needs full text |
| Silver nanowire/PVA 5 wt%, 26 dB | Zeng et al., Small 13, 1701388, 2017 | Paper is MWCNT/waterborne polyurethane, not AgNW/PVA. | Wrong material; find real source or drop |
| rGO film, 8.5 um, 20 dB, sigma 2.5e4 S/m | Shen et al., Adv. Funct. Mater. 2014 | About 8.4 um and about 20 dB confirmed. Sigma not confirmed. | Needs full text for sigma |
| Cu, Al foil 0.1 mm at 1 MHz (78, 74 dB) | Schulz, Plantz & Brush, IEEE TEMC 1988 | Abstract describes calculated performance of copper and iron shields, not measurements. File's R/A split does not match theory. | Not shown to be a measurement; needs full text or drop |
| Ag foil 0.1 mm at 1 MHz, 80 dB | Ott, EMC Engineering, 2009 | Textbook; not checked. | Needs page reference or drop |
| SS304 0.5 mm at 1 MHz, 50 dB | Chung, J. Mater. Eng. Perform. 2000 | Not checked. | Needs full text |
| Short carbon fibre/epoxy 5, 15, 25 wt% | Jana et al., ACS AMI 6, 12588, 2014 | Could not locate a paper matching this citation or DOI. | Citation unconfirmed |
| CFRP unidirectional, 15 dB | Chung, Carbon 39, 279, 2001 | Not checked. | Needs full text |
| Graphene foam/PDMS, 30 dB | Song et al., Small 10, 4000, 2014 | Per the manuscript's own reference list this paper is about graphene papers, not graphene foam/PDMS. | Likely wrong citation; needs check |
| ITO glass 28 GHz; conductive paint 2.4 GHz | Wanasinghe et al., Nanomaterials 2020 (review) | Secondary source; not checked. | Needs primary source |
