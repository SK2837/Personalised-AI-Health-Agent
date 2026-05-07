"""
PHAI - curated SNP panel.

Ten well-studied, lifestyle-relevant SNPs. Each entry has:
  - rsid, gene, alleles (major, minor)
  - minor_allele_freq: published European-population MAF (gnomAD / 1000G order of magnitude)
  - genotype_meanings: short explanation per genotype
  - trait_summary: one-paragraph plain-English description for the agent
  - lifestyle_implications: one-paragraph "what to do about it" for the coach
  - clinvar_significance: classification (mostly polymorphism / risk-factor)
  - citation_url: dbSNP page for traceability

Single source of truth - both the snp_reference SQLite table and the
Domain Expert agent read from here.
"""

from __future__ import annotations


PANEL: list[dict] = [
    {
        "rsid": "rs1801260",
        "gene": "CLOCK",
        "alleles": ("T", "C"),  # (major, minor)
        "minor_allele_freq": 0.30,
        "genotype_meanings": {
            "TT": "Earlier chronotype tendency.",
            "TC": "Intermediate chronotype.",
            "CC": "Later chronotype tendency, often shorter habitual sleep.",
        },
        "trait_summary": (
            "CLOCK helps regulate the body's circadian rhythm. The minor C "
            "allele is associated with later chronotype (evening preference) "
            "and slightly shorter habitual sleep duration."
        ),
        "lifestyle_implications": (
            "C-carriers with sleep complaints often benefit from 20+ minutes "
            "of morning daylight, a fixed wake time even on weekends, and "
            "finishing caffeine intake before noon to anchor the circadian "
            "rhythm."
        ),
        "clinvar_significance": "polymorphism",
        "citation_url": "https://www.ncbi.nlm.nih.gov/snp/rs1801260",
    },
    {
        "rsid": "rs228697",
        "gene": "PER3",
        "alleles": ("C", "G"),
        "minor_allele_freq": 0.10,
        "genotype_meanings": {
            "CC": "Typical circadian phasing.",
            "CG": "Slight earlier-phasing tendency.",
            "GG": "Stronger earlier-phasing tendency reported in some studies.",
        },
        "trait_summary": (
            "PER3 is another core circadian-clock gene. The G allele has been "
            "associated in some cohorts with earlier sleep timing and "
            "better tolerance of morning awakenings."
        ),
        "lifestyle_implications": (
            "G-carriers tend to do well with morning workouts and early "
            "starts. CC-carriers should avoid forcing extreme early routines "
            "and instead align activity with their natural mid-morning peak."
        ),
        "clinvar_significance": "polymorphism",
        "citation_url": "https://www.ncbi.nlm.nih.gov/snp/rs228697",
    },
    {
        "rsid": "rs1042713",
        "gene": "ADRB2",
        "alleles": ("A", "G"),
        "minor_allele_freq": 0.40,
        "genotype_meanings": {
            "AA": "Arg16 homozygous - typical bronchodilator and exercise response.",
            "AG": "Heterozygous.",
            "GG": "Gly16 homozygous - altered beta-2 agonist response, sometimes faster bronchodilation desensitisation.",
        },
        "trait_summary": (
            "ADRB2 codes for the beta-2 adrenergic receptor, important in "
            "bronchodilation and the cardiovascular response to exercise. "
            "The G (Gly16) allele can alter how the body responds to "
            "beta-2 agonists and to high-intensity exercise."
        ),
        "lifestyle_implications": (
            "GG-carriers may benefit from longer warm-ups, more gradual "
            "intensity ramps, and (if asthmatic) discussion with a clinician "
            "about inhaler choice. AA-carriers typically tolerate standard "
            "high-intensity protocols well."
        ),
        "clinvar_significance": "polymorphism",
        "citation_url": "https://www.ncbi.nlm.nih.gov/snp/rs1042713",
    },
    {
        "rsid": "rs1815739",
        "gene": "ACTN3",
        "alleles": ("C", "T"),
        "minor_allele_freq": 0.42,
        "genotype_meanings": {
            "CC": "RR - functional alpha-actinin-3 in fast-twitch muscle, power-oriented.",
            "CT": "RX - one functional copy.",
            "TT": "XX - no functional alpha-actinin-3, endurance-oriented.",
        },
        "trait_summary": (
            "ACTN3 (the so-called 'sprinter gene') produces a protein in "
            "fast-twitch muscle fibres. The T (X) allele is a stop codon - "
            "TT-carriers produce no functional protein and are over-represented "
            "in endurance athletes; CC-carriers are over-represented in "
            "power and sprint athletes."
        ),
        "lifestyle_implications": (
            "TT-carriers usually respond well to endurance training (longer "
            "runs, cycling, swimming) and may need extra warm-up before "
            "explosive efforts. CC-carriers tend to respond well to power "
            "and sprint training."
        ),
        "clinvar_significance": "polymorphism",
        "citation_url": "https://www.ncbi.nlm.nih.gov/snp/rs1815739",
    },
    {
        "rsid": "rs762551",
        "gene": "CYP1A2",
        "alleles": ("A", "C"),
        "minor_allele_freq": 0.32,
        "genotype_meanings": {
            "AA": "Fast caffeine metaboliser.",
            "AC": "Intermediate metaboliser.",
            "CC": "Slow caffeine metaboliser - caffeine lingers longer.",
        },
        "trait_summary": (
            "CYP1A2 codes for the liver enzyme that metabolises caffeine. "
            "The A allele encodes the faster-acting form. Slow metabolisers "
            "(C-carriers, especially CC) clear caffeine more slowly and tend "
            "to feel its effects - including sleep disruption - for longer."
        ),
        "lifestyle_implications": (
            "Slow metabolisers should keep caffeine modest (≤ 200-300 mg/day) "
            "and finish their last cup at least 8-10 hours before bed. Fast "
            "metabolisers tolerate later caffeine but should still cap intake "
            "and watch sleep response."
        ),
        "clinvar_significance": "polymorphism",
        "citation_url": "https://www.ncbi.nlm.nih.gov/snp/rs762551",
    },
    {
        "rsid": "rs1801133",
        "gene": "MTHFR",
        "alleles": ("G", "A"),  # G=C in the C677T literature; using forward strand
        "minor_allele_freq": 0.36,
        "genotype_meanings": {
            "GG": "Typical MTHFR enzyme activity.",
            "GA": "Mildly reduced enzyme activity.",
            "AA": "Significantly reduced enzyme activity (~30% of typical).",
        },
        "trait_summary": (
            "MTHFR (the C677T variant) influences how the body converts "
            "folate into its active form. AA-carriers (TT in the literature) "
            "have reduced enzyme activity, which can affect homocysteine "
            "levels and folate utilisation in some individuals."
        ),
        "lifestyle_implications": (
            "Reduced-activity carriers often benefit from folate-rich foods "
            "(leafy greens, legumes, citrus) and should discuss methylated "
            "folate forms with a clinician if homocysteine is elevated. "
            "Moderate, regular B-vitamin intake supports normal methylation."
        ),
        "clinvar_significance": "risk-factor",
        "citation_url": "https://www.ncbi.nlm.nih.gov/snp/rs1801133",
    },
    {
        "rsid": "rs9939609",
        "gene": "FTO",
        "alleles": ("T", "A"),
        "minor_allele_freq": 0.45,
        "genotype_meanings": {
            "TT": "Lower-risk genotype for elevated BMI.",
            "TA": "Intermediate.",
            "AA": "Higher reported BMI tendency, possibly stronger appetite drive.",
        },
        "trait_summary": (
            "FTO is the most replicated common variant linked to BMI. "
            "A-carriers (especially AA) show modestly higher average BMI "
            "and, in some studies, higher hunger drive and reduced satiety "
            "after meals."
        ),
        "lifestyle_implications": (
            "A-carriers tend to respond well to high-protein, high-fibre "
            "meals, consistent meal timing, and mindful-eating practices. "
            "Regular structured exercise blunts much of the effect."
        ),
        "clinvar_significance": "risk-factor",
        "citation_url": "https://www.ncbi.nlm.nih.gov/snp/rs9939609",
    },
    {
        "rsid": "rs429358",
        "gene": "APOE",
        "alleles": ("T", "C"),
        "minor_allele_freq": 0.14,
        "genotype_meanings": {
            "TT": "Not e4 at this position - lower cardiovascular and Alzheimer's risk.",
            "TC": "One e4 allele - moderately elevated risk.",
            "CC": "Two e4 alleles - elevated cardiovascular and Alzheimer's risk.",
        },
        "trait_summary": (
            "APOE is central to lipid metabolism and brain health. The C "
            "allele at rs429358 (combined with rs7412) defines the e4 "
            "haplotype, which is associated with higher LDL cholesterol "
            "and elevated late-onset Alzheimer's risk."
        ),
        "lifestyle_implications": (
            "C-carriers benefit from a diet lower in saturated fat, richer "
            "in omega-3 (oily fish, walnuts, flax), and from regular cardio. "
            "Sleep quality and cognitive engagement also matter - APOE e4 "
            "is a risk factor, not a destiny."
        ),
        "clinvar_significance": "risk-factor",
        "citation_url": "https://www.ncbi.nlm.nih.gov/snp/rs429358",
    },
    {
        "rsid": "rs4680",
        "gene": "COMT",
        "alleles": ("G", "A"),
        "minor_allele_freq": 0.50,
        "genotype_meanings": {
            "GG": "Val/Val - faster dopamine clearance ('warrior'): better stress tolerance, more variable focus.",
            "GA": "Val/Met - intermediate.",
            "AA": "Met/Met - slower dopamine clearance ('worrier'): better sustained focus, more sensitive to stress.",
        },
        "trait_summary": (
            "COMT breaks down dopamine in the prefrontal cortex. The A (Met) "
            "allele encodes a slower-acting enzyme, which is associated with "
            "better sustained attention but greater sensitivity to acute "
            "stress (the so-called 'worrier vs warrior' polymorphism)."
        ),
        "lifestyle_implications": (
            "AA-carriers benefit most from stress-management practices - "
            "mindfulness, breathwork, paced workloads. GG-carriers often "
            "thrive under cognitive load and may need novelty and complexity "
            "to maintain focus."
        ),
        "clinvar_significance": "polymorphism",
        "citation_url": "https://www.ncbi.nlm.nih.gov/snp/rs4680",
    },
    {
        "rsid": "rs6265",
        "gene": "BDNF",
        "alleles": ("C", "T"),
        "minor_allele_freq": 0.20,
        "genotype_meanings": {
            "CC": "Val/Val - typical activity-dependent BDNF secretion.",
            "CT": "Val/Met - reduced activity-dependent secretion.",
            "TT": "Met/Met - further reduced; reported associations with memory and stress reactivity.",
        },
        "trait_summary": (
            "BDNF supports neuron growth and synaptic plasticity. The T (Met) "
            "allele alters activity-dependent secretion and has been linked "
            "in some studies to differences in memory consolidation and "
            "stress recovery."
        ),
        "lifestyle_implications": (
            "Aerobic exercise reliably raises BDNF and is especially helpful "
            "for T-carriers. Adequate sleep, learning new skills, and "
            "managing chronic stress also support BDNF function."
        ),
        "clinvar_significance": "polymorphism",
        "citation_url": "https://www.ncbi.nlm.nih.gov/snp/rs6265",
    },
]

# Convenience lookups
PANEL_BY_RSID: dict[str, dict] = {snp["rsid"]: snp for snp in PANEL}
RSIDS: list[str] = [snp["rsid"] for snp in PANEL]


def get_snp(rsid: str) -> dict | None:
    """Return the panel entry for a given rsid, or None."""
    return PANEL_BY_RSID.get(rsid)
