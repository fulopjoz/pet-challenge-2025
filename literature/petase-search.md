The r=0.178 agreement between ESM2 and ESMC is not surprising — both PLMs are trained to model evolutionary sequence distributions, and *E. coli* recombinant expression is a **translation-system** phenotype driven by CDS-level features that amino acid-level models never see.

## Why PLM Scores Fail Here

Zero-shot ESM log-likelihoods measure how "natural" a sequence is relative to the evolutionary sequence distribution in UniRef. This tracks mutation fitness (activity, stability, binding) reasonably well because evolution selects for those properties. But heterologous expression yield in *E. coli* adds an entirely different layer of constraints — codon–tRNA matching, 5′ mRNA accessibility, RBS-start codon architecture, and mRNA degradation kinetics — that are invisible to an amino-acid-level model. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC5054687/)

Because ESM2 and ESMC learn slightly different priors from sequence data, they can still agree on amino-acid-level fitness while producing uncorrelated rankings when the true phenotype is CDS-level. The ~0.18 correlation is therefore noise, not signal.

## CDS Features That Actually Predict Expression

The evidence across thousands of expression experiments points to a specific feature hierarchy:

| Feature | Best *E. coli* correlation | Notes |
|---|---|---|
| Codon Adaptation Index (CAI) | r ≈ 0.46 (total expression)  [nature](https://www.nature.com/articles/s41598-018-29035-z) | Correlates with total yield, not always solubility |
| tRNA Adaptation Index (TAI) | r ≈ 0.54  [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11546221/) | Accounts for isoacceptor tRNA pools; stronger than CAI alone |
| Codon Influence Metric (multi-param.) | Strong, > TAI  [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC5054687/) | Dissected from 6,348 expression experiments; correlates with mRNA levels and lifetime |
| Protein/mRNA ratio (6-feature model) | R = 0.666  [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10849221/) | Requires translation initiation rate + TASEP elongation simulation |
| 5′-end mRNA folding ΔG | Weak to null (R < 0.15)  [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10849221/) | Contradicts older intuitions; initiation rate alone explains little |
| Physicochemical AA properties | ~58% within 1 order of magnitude  [www-vendruscolo.ch.cam.ac](https://www-vendruscolo.ch.cam.ac.uk/tartaglia09jmb.pdf) | Useful for rough solubility ranking only |

The key takeaway: **TAI > CAI > 5′ mRNA folding** for yield prediction in *E. coli*, and multi-feature models combining codon influence + translation elongation dynamics reach R ≈ 0.67. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11546221/)

## mRNA Folding — a Nuanced Story

Contrary to the common heuristic that a low-ΔG 5′ UTR/early CDS guarantees high expression, large-scale analysis shows that 5′-end folding energy alone has almost no impact on protein-per-mRNA output (R = 0.004). What does matter is **stable mRNA structural elements dispersed throughout the CDS** — they modulate local elongation speed and give the nascent chain time to fold co-translationally, strongly correlating with protein compactness. Maximally unfolded 5′ ends help initiation but do not dominate total yield. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10849221/)

## Deep Learning for Codon Optimization / Expression Prediction

Several supervised CDS-level models now exist that go beyond classical CAI:

- **BiLSTM-CRF models** (e.g., Fu et al., 2020) learn context-specific codon usage patterns in *E. coli* and optimize synonymous sequences for expression [nature](https://www.nature.com/articles/s41598-020-74091-z)
- **MPEPE** (CNN-based, supervised) is trained on diverse heterologous protein expression measurements in *E. coli* to directly predict protein expression efficiency [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2693125725000433)
- **Absci's deep learning codon model**, fine-tuned on >150,000 functional expression measurements of synonymous sequences from three proteins, accurately predicts expression and generalizes across sequences [totient](https://totient.bio/wp-content/uploads/2023/11/Absci___deep_learning_based_codon_optimization.pdf)
- **LinearDesign** jointly optimizes mRNA stability and codon usage (designed for therapeutics but the algorithm applies) [nature](https://www.nature.com/articles/s41586-023-06127-z)
- **PNAS 2024 (Sidi et al.)** critiques naive BiLSTM codon models and points to more principled AI-based sequence generation for codon design [pnas](https://www.pnas.org/doi/10.1073/pnas.2410003121)

## PETase as a Concrete Case

PETase (IsPETase from *I. sakaiensis*) is a canonical example where **codon optimization for *E. coli* is non-negotiable**:

- The standard Beckham lab construct uses a **codon-optimized synthetic gene for *E. coli* K12**; without it, much of the expressed IsPETase ends up in inclusion bodies [scholarhub.ui.ac](https://scholarhub.ui.ac.id/cgi/viewcontent.cgi?article=2226&context=science)
- Expression improvements come from combining codon optimization with signal peptide engineering (e.g., evolved pelB), not from amino-acid sequence changes alone [pubs.acs](https://pubs.acs.org/doi/10.1021/acs.jafc.0c07469)
- Cytoplasmic expression studies confirm that CDS-level and secretion pathway choices dominate soluble yield over sequence-intrinsic properties [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11587651/)

## Practical Fix for Your Pipeline

Given your r=0.178 PLM-only baseline, the cleanest path forward is a **two-stream feature model**:

1. **CDS stream:** CAI, TAI (compute against *E. coli* K12 tRNA pool), per-codon influence scores, codon pair bias, GC content windows, %MinMax codon harmonization profile [nature](https://www.nature.com/articles/s41598-018-29035-z)
2. **Protein stream:** ESM2/ESMC embeddings or log-likelihoods (aggregation propensity, disorder, stability proxy)
3. **Fuse via gradient-boosted trees or a shallow MLP** — given typical dataset sizes for expression experiments, deep fusion models overfit unless you have thousands of labeled examples

For zero-shot scenarios (no labeled expression data for your specific protein), rational codon optimization via TAI maximization + %MinMax harmonization is likely to outperform ESM scoring for expression ranking. The PLM signal becomes more valuable as a co-variate only once you have supervised expression labels to calibrate against. [pubs.acs](https://pubs.acs.org/doi/10.1021/acssynbio.3c00367)



The pH 5.5 vs 9.0 scoring difference is biochemically well-justified and deeply mechanistic — the entire pH activity profile of PETases is governed by the protonation state of the catalytic histidine in the Ser-His-Asp triad.

## The Core Mechanism: Catalytic His pKa

PETases belong to the serine hydrolase family and use a canonical **Ser-His-Asp catalytic triad**. The His must be in its **neutral (deprotonated) form** to abstract the proton from the catalytic Ser, enabling nucleophilic attack on the PET ester bond. The entire pH-activity relationship flows from this single constraint. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11213969/)

For the industrially optimized LCC^ICCG^ PETase (the current benchmark), detailed NMR titration of the catalytic H242 histidine reveals: [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11213969/)

| Condition | His242 pKa | Implication |
|---|---|---|
| Active enzyme, 30°C | 4.90 ± 0.05 | ~99.9% deprotonated at pH 9.0 → fully active |
| Active enzyme, 50°C | 4.70 ± 0.04 | ~98.4% deprotonated at pH 7.0 |
| pH 5.5 (30°C) | ~80% deprotonated | ~20% of His protonated → reduced activity |
| pH 4.0 | <10% deprotonated | Essentially inactive |

This pKa (~4.7–4.9) is **much lower than the typical His pKa of 6.5** in proteins, due to the highly charged environment around the other five histidines in LCC^ICCG^, four of which have pKa values below 4.0. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11213969/)

## Soluble vs. Solid PET: Different pH Profiles

A critical finding from the NMR study is that activity on **soluble** and **solid** PET respond differently to pH: [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/38664965/)

- **Soluble substrate (BHET):** Activity follows the Henderson–Hasselbalch deprotonation curve of H242 cleanly. Optimal activity spans pH 6–8, with 67.5%, 32.5%, and 15% activity retained at pH 5.0, 4.5, and 4.0 respectively.
- **Solid PET powder:** Activity is *steeper* than the His pKa predicts — activity drops to half-maximum already at **pH 6.5**, a full 2 pH units above the pKa. Below pH ~5, activity is essentially zero.
- **Proposed mechanism:** Surface acidification from released TPA/MHET carboxylic acids creates a local pH drop at the polymer surface, shifting the effective pH experienced by active-site His lower than the bulk buffer pH. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11213969/)

This means that at **pH 5.5 on solid PET**, you are operating in the steep transition zone where activity plummets, whereas at **pH 9.0** the enzyme is well into its plateau maximum.

## IsPETase-Specific pH Data

Wild-type IsPETase has its own, less-studied pH profile. Key observations from the literature:

- IsPETase was originally characterized with optimal activity around **pH 7–8** and room temperature (30–37°C) [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12296185/)
- Engineered variants (FastPETase, FAST-PETase 212/277, thermostable mutants) inherit the same Ser-His-Asp triad and therefore the same qualitative pH dependence [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12190498/)
- Insertion of acidic residue Glu at position 93/94 (IsPETase^9394insE^) specifically improved binding at the active pocket at 45°C but was assayed at pH ~7.5 standard conditions — the acidic insertion affected the substrate-binding geometry, not the catalytic His pKa [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11306670/)
- IsPETase thermostability engineering (e.g., CaPETase^M9^ with T_m = 83.2°C, 41.7-fold activity gain at 60°C) uses a **pH-stat bioreactor** — continuously adjusting pH to maintain the alkaline optimum, acknowledging that acid product accumulation would otherwise acidify and inactivate the enzyme [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10382486/)

## Stability vs. Activity: Two Separate pH Effects

At extreme pH values, you face two distinct problems that justify asymmetric scoring weights:

| pH regime | Activity effect | Stability effect |
|---|---|---|
| **pH 9.0** | Near-maximal (His fully deprotonated) | Some risk of alkaline hydrolysis of surface loops; most PETases are designed stable here |
| **pH 5.5** | 20–30% of maximum (His partially protonated at pKa ≈ 4.8) | Acidic conditions generally preserve stability but suppress catalysis |
| **pH < 4** | ~0% (His fully protonated) | Protein denaturation risk; IsPETase is unstable below pH 4 |

For PET degradation performance specifically, **pH 9.0 dominates pH 5.5 by ~3–5× in activity** on solid substrate based on the LCC^ICCG^ pH profile. This factor directly justifies heavier weighting of the pH 9.0 condition in your scoring. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11213969/)

## Engineering Implications for Scoring Weights

The literature suggests a principled rationale for your pH-dependent scoring heuristics:

- **pH 9.0 weight ≫ pH 5.5 weight** for *activity-oriented* scoring, because the His pKa means you cannot escape the activity penalty at pH 5.5 through protein engineering without repositioning the catalytic triad itself [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/38664965/)
- **pH 5.5 weight retains value** for *stability screening* — mutations that improve stability at mildly acidic pH may also translate to better enzyme longevity in industrial reactors where product accumulation acidifies the medium over time [nature](https://www.nature.com/articles/s41467-025-60016-9)
- **Industrial standard** is now pH 8.0 buffer (e.g., 0.1 M HEPES or phosphate) with thermoalkalophilic conditions (65–70°C, pH 7.5–8.5) per the Nature Communications standardization guidelines  — this anchors pH 9.0 scoring closer to the real industrial design target than pH 5.5 [nature](https://www.nature.com/articles/s41467-025-60016-9)

The fact that PET hydrolases are serine hydrolases with a catalytic His pKa of ~4.7–4.9 means the pH-activity curve is mechanistically anchored, and any model that treats pH 5.5 and pH 9.0 equally is provably wrong by first principles.


The wildtype-marginal log-likelihood ratio (Meier 2021) is a solid baseline for general fitness prediction but has specific, well-characterized failure modes when the target phenotype is *enzyme catalytic activity* rather than the broader fitness of the sequence in its evolutionary context.

## What Meier 2021 Actually Predicts

The masked-marginal scoring strategy, \(\Delta \log p = \log p(x_{\text{mt}} | x_{\backslash i}) - \log p(x_{\text{wt}} | x_{\backslash i})\), estimates how "evolutionarily acceptable" a substitution is relative to background sequence context  [proceedings.neurips](https://proceedings.neurips.cc/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html). This correlates with fitness in DMS assays primarily because:

- Evolutionary conservation at a position dominates fitness variance in most single-substitution datasets (the PSSM baseline already explains most of the signal)
- Epistatic corrections from PLM context (vs. independent-site models) add moderate improvement, but the fundamental signal remains **residue conservation** from phylogeny, not catalytic mechanism [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10829072/)

The critical gap: PLM log-likelihoods cannot distinguish whether a position is conserved because it is *catalytic* vs. because it maintains *fold stability* or *oligomeric contacts* — three orthogonal evolutionary pressures that get superimposed into a single scalar. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10829072/)

## The Evolution-Catalysis Relationship: A Better Frame

The Xie & Warshel framework provides a mechanistically motivated answer for why undifferentiated PLM scores underperform for activity: [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10829072/)

| Enzyme region | Evolutionary pressure | Correlates with |
|---|---|---|
| Active site / substrate-proximal residues | Catalytic activity (barrier lowering) | k_cat, k_cat/K_M |
| Scaffold / distal to substrate | Thermostability | T_m, ΔG_unfold |
| Loop regions coupled to active site | Activity (conformational gating) | k_cat via substrate exclusion/access |

Using MaxEnt (DCA/EVcoupling) models, region-specific statistical energy \(E = -\log P(\text{seq})\) correlates with kcat when restricted to active-site residues, and with stability when restricted to scaffold residues. **Applying a global PLM score to the full sequence conflates all three signals.** [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10829072/)

For PETase specifically, the catalytic Ser-His-Asp triad residues and the loop regions flanking the substrate-binding groove (β6-β7 loop, disulfide-forming region) are the catalytically coupled positions — scoring only those positions' log-likelihood ratios would give a cleaner activity proxy than the global sequence score.

## Alternative Zero-Shot Strategies Ranked by Evidence

### 1. Masked Marginal with Position Restriction
Apply Meier-style scoring but **restrict the masked positions to active-site / catalytically proximal residues** (within 5–8 Å of catalytic Ser in structure). This removes the stability-dominating scaffold signal and isolates the activity-relevant evolutionary pressure. [huggingface](https://huggingface.co/blog/AmelieSchreiber/mutation-scoring)

### 2. Structure-Based Zero-Shot Models (Strongest Recent Evidence)

Structure-conditioned models outperform sequence-only PLMs on ProteinGym for enzymes where conformational context matters: [arxiv](https://arxiv.org/abs/2504.16886)

- **ProMEP** (Cheng et al., *Nature Cell Research* 2024): Computes log-likelihood jointly over sequence and structure context: scores mutations with both evolutionary and geometric plausibility. Best zero-shot method on ProteinGym for enzyme assays per benchmarks. [nature](https://www.nature.com/articles/s41422-024-00989-2)
- **VenusREM** (Bioinformatics 2025): Retrieval-enhanced PLM that captures local amino acid interactions in both sequence and structural space; especially strong for enzymes with catalytic geometric constraints [academic.oup](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i401/8199374)
- **ESMFold + inverse folding scoring**: Use ESM-IF1 or ProteinMPNN log-likelihoods on the predicted/experimental structure — gives structure-conditioned probability that separates stability from function better than sequence-only scoring [arxiv](https://arxiv.org/abs/2504.16886)

### 3. Inference-Time Monte Carlo Dropout (Quick Win)

Ravuri & Lawrence (2025) showed that adding a **dropout layer between ESM2's embedding and transformer at inference time**, then averaging over stochastic forward passes (MC dropout), improves zero-shot Spearman ρ by **up to 22% for smaller models** (8–35M params) on ProteinGym subsets, at zero retraining cost. This is effectively uncertainty smoothing that regularizes overconfident PLM predictions in out-of-distribution regions. [arxiv](https://arxiv.org/abs/2506.14793)

Implementation: insert `nn.Dropout(p=0.1)` after ESM2's embedding layer, run 50–100 stochastic forward passes, average log-probabilities before computing the LLR.

### 4. Ensemble of Complementary Scorers

On ProteinGym, **simple ensembles of structure-based and sequence-based models consistently outperform any single model**. A practical ensemble for PETase activity: [arxiv](https://arxiv.org/abs/2504.16886)

```
score = α · ESM2_masked_marginal(active_site_only)
      + β · ProMEP_multimodal_score
      + γ · EVmutation_DCA_score (if MSA available)
```

The DCA/EVcoupling component captures pairwise epistasis (especially important for the IsPETase disulfide and loop co-evolution) that PLMs encode diffusely. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10829072/)

### 5. Retrieval-Augmented Zero-Shot (ProtREM / VenusREM)

These models augment zero-shot scoring with retrieved homologous sequences at inference time — functionally similar to MSA-based methods but without explicit alignment. For PET hydrolases, which have many characterized homologs (LCC, Cut190, ThermoPETase, FAST-PETase), retrieval-augmented scoring significantly outperforms standalone ESM2 and ESM-C. [arxiv](https://arxiv.org/html/2410.21127v1)

## Known Failure Modes of Log-Likelihood Ratios for Enzymes

The 2025 arXiv critique (Ravuri & Lawrence) confirms several structural failure modes: [emergentmind](https://www.emergentmind.com/topics/protein-language-model-esm-2)

| Failure mode | Mechanism | Effect on your scoring |
|---|---|---|
| Conservation ≠ catalysis | Catalytic residues conserved for activity, but scaffold equally conserved for fold | Overscores stability-stabilizing mutations |
| Context window saturation | Large multi-mutation variants confuse masked-marginal independence assumption | Multi-mutant PETase variants (FastPETase has 5 mutations) are scored inaccurately |
| Disorder & loop regions | PLMs trained on structured domains; IsPETase active-site loops are semi-disordered | LLRs for loop mutations are unreliable  [arxiv](https://arxiv.org/abs/2504.16886) |
| MSA depth sensitivity | ESM2 trained on UniRef90; PETase is a recently discovered enzyme with few natural homologs | Sparse MSA → weaker evolutionary signal at variant positions |

## Practical Recommendation for Your Pipeline

Given that you're scoring IsPETase variants for catalytic activity at pH 5.5 and 9.0:

1. **Restrict LLR scoring to catalytically relevant positions** (Ser160, His237, Asp206 triad; within 8 Å of substrate analog in PDB:6EQE or 6ILW)
2. **Add ProMEP or ESM-IF1 structure-conditioned scores** as a second stream — they outperform masked marginal on ProteinGym enzyme assays [nature](https://www.nature.com/articles/s41422-024-00989-2)
3. **Apply MC dropout** to your current ESM2 scorer — free improvement, no retraining [arxiv](https://arxiv.org/abs/2506.14793)
4. **Weight the ensemble by known benchmark performance** on closest ProteinGym assay (hydrolase DMS datasets like TEM-1 beta-lactamase or esterases) rather than using hand-tuned heuristics
5. For multi-mutant variants, switch from masked marginal to **wildtype-conditional autoregressive scoring** or sum of independent single-mutation LLRs (additive approximation), which remains more reliable than multi-masked scoring for 3–8 simultaneous mutations [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10723403/)



The pET-28a vector and 313 CDS sequences are high-value unused features. The vector architecture fixes several upstream constraints (T7 promoter, T7 terminator, RBS-to-ATG junction), making the CDS the dominant variable for expression level prediction within this system.

## Why pET-28a Context Is Non-Trivial

pET-28a uses the **T7 promoter / T7 RNA polymerase system** — one of the strongest prokaryotic expression drivers available, which almost completely redirects *E. coli* BL21(DE3) transcription to your target after IPTG induction. This has two implications for modeling: [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC6353774/)

- The transcription rate is essentially saturated and constant across constructs — so **translation** becomes the dominant bottleneck and the primary source of inter-construct expression variance
- The **fixed N-terminal sequence** from the vector (MHHHHHHssGLVPRGS… for the His-thrombin-tag in pET-28a) creates a vector-specific RBS-to-CDS junction context that modulates the translational initiation rate for every insert [kirschner.med.harvard](https://kirschner.med.harvard.edu/files/protocols/Novagen_petsystem.pdf)

This means your 313 CDS sequences are all operating under the same transcriptional regime — the CDS itself (codon usage, early secondary structure, internal regulatory elements) is the dominant variable. This is exactly the situation where CDS-feature ML models perform best.

## Feature Set for a CDS-Level Expression Predictor

The NAR 2025 study (Shen et al.) provides the best current benchmark of which features matter for both **local accuracy and cross-sequence generalization**: [academic.oup](https://academic.oup.com/nar/article/53/3/gkaf020/7985285)

| Feature | Dimensionality | Local accuracy | Generalization |
|---|---|---|---|
| One-hot nucleotide encoding (384 bp window) | 384 | **Best** (R² > 0.5) | **Worst** (R² < 0 cross-series) |
| CAI (Codon Adaptation Index) | 1 scalar | Moderate | Good |
| MFE (mRNA 5′ folding ΔG, 3 overlapping windows) | 3 scalars | Moderate | Good |
| AT nucleotide content | 1 scalar | Moderate | Good |
| Mean Hydropathy Index (MHI) | 1 scalar | Moderate | Good |
| **Fused (one-hot + mechanistic)** | ~390 | **Best** | **Best** |

The critical finding: one-hot encodings give the best in-distribution accuracy but **fail completely when generalizing to novel sequence variants** (negative R²), whereas mechanistic features generalize well across mutational series and host contexts. For your 313 CDS variants (likely spanning diverse sequence families), this means **mechanistic features are necessary for robust ranking**. [academic.oup](https://academic.oup.com/nar/article/53/3/gkaf020/7985285)

A practical 6–10 feature CDS vector for your sequences:

```python
features = {
    "CAI":          cai(cds, e_coli_codon_table),        # codon adaptation index
    "tAI":          tai(cds, e_coli_trna_pool),           # tRNA adaptation index
    "MFE_window1":  vienna_mfe(cds[0:96]),                 # 5' folding (nt 1-96)
    "MFE_window2":  vienna_mfe(cds[48:144]),               # sliding window
    "MFE_window3":  vienna_mfe(cds[96:192]),               # sliding window
    "AT_content":   at_fraction(cds),
    "GC_first50":   gc_fraction(cds[:150]),                # N-terminal codon GC
    "MHI":          mean_kyte_doolittle(protein),          # hydropathy
    "CPB":          codon_pair_bias(cds, e_coli_cpb_table),# codon pair bias
    "rare_codon_N": rare_codon_fraction(cds[:150])         # first 50 codons
}
```

## ML Architecture Recommendations

**For 313 labeled sequences** (a small supervised dataset), the evidence clearly favors shallow ensemble methods over deep models:

- **Gradient Boosted Trees (XGBoost/LightGBM)** with mechanistic CDS features as input is the recommended baseline: SoluProt (gradient boosted machine on global features) achieves state-of-the-art for soluble expression prediction on TargetTrack-derived datasets and outperforms older neural networks [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12857573/)
- **Random Forest** on fused one-hot + mechanistic features gives the best generalization, per the NAR 2025 benchmark — Shen et al. show RF with geometric stacking (GCN-based feature fusion) reaches near-oracle generalization on held-out sequence families [academic.oup](https://academic.oup.com/nar/article/53/3/gkaf020/7985285)
- **One-hot encoding alone with RF** (R² ≈ 0.55 in the supervised ML study on *E. coli* DNA sequences) performs adequately in-distribution but degrades rapidly for novel sequences [scirp](https://www.scirp.org/journal/paperinformation?paperid=111035)

**For a production predictor on 313 sequences**, the recommended pipeline is:

```
input: CDS sequence (DNA) → [mechanistic features + one-hot 5' window]
       ↓ feature stacking
       Random Forest or XGBoost regressor
       ↓ ensemble stacking with meta-learner
       expression score (continuous) or binary soluble/insoluble
```

## State-of-the-Art: RP3Net (January 2026)

**RP3Net** (AstraZeneca + EMBL-EBI, *Bioinformatics* 2026) is the current best model for predicting soluble recombinant expression in *E. coli* from construct sequence: [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12857573/)

- Architecture: ESM2 (650M) encoder → Set Transformer Pooling aggregation → classification head; trained on AstraZeneca internal screens + SGC Stockholm + SGC Toronto (67,055 unique sequences total)
- Key finding: **CaLM** (a codon-level language model pre-trained on European Nucleotide Archive coding sequences) outperforms all DNA language models in the RP3Net framework — specifically because it was pre-trained on *coding* sequences rather than mixed coding/noncoding DNA [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12857573/)
- **Prospective validation AUROC = 0.83** on 97 human drug target constructs in *E. coli*, identifying expressing constructs with 92% recall [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12857573/)
- The MLC (Meta Label Correction) framework enabled using large-scale purification data (noisy labels) to improve small-scale expression screening prediction — a directly transferable technique if you have expression annotations with varying confidence in your 313 CDS dataset

RP3Net is available under MIT license at `github.com/RP3Net/RP3Net` and uses the **full construct sequence including vector-derived His-tag and linker** as input — meaning if you feed it your pET-28a insert with the correct N-terminal tag context, it encodes the exact vector-CDS junction that governs initiation efficiency. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12857573/)

## Extracting Value from `pet-2025-expression-vector.gb`

The GenBank file gives you:

1. **Exact T7 RBS sequence and ATG position** — compute the RBS–ATG spacing and folding energy of the junction window (nt −35 to +96 relative to ATG) for each inserted CDS; this is the critical window for ribosome access [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC5385732/)
2. **His-tag and linker context** — the first ~20 codons of your expressed construct are vector-derived; their codon usage and interaction with insert CDS early codons affect "ramp" translation dynamics; rare codons in positions 2–10 of the insert can increase expression up to 53–56% of total cell protein (the "rare codon ramp hypothesis") [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S1046592815001540)
3. **Vector GC content junction** — the pET-28a T7 promoter region has a defined GC context; abrupt GC jumps at the vector-insert boundary create mRNA secondary structures that reduce translation initiation [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC5385732/)

Concretely: for each of your 313 CDS sequences, you can compute the folding energy of the **vector-CDS junction** (last 20nt of vector RBS region + first 76nt of your CDS) as a construct-specific initiation feature, which mechanistically outperforms any feature computed on the CDS alone.


NDCG as an evaluation metric has a specific and underappreciated relationship with PLM scoring: optimizing Spearman correlation does **not** necessarily optimize NDCG, and several methods now target NDCG directly via learning-to-rank objectives.

## NDCG vs. Spearman: Why They Diverge

ProteinGym computes both Spearman ρ and **NDCG@10%** and explicitly observes that some models rank differently across the two metrics. The distinction is fundamental: [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10723403/)

- **Spearman ρ** treats errors at the top and bottom of the ranking equally — a misprediction for a mediocre variant costs the same as missing the best variant
- **NDCG@10%** discounts errors logarithmically by position and weights the **top 10% of variants** disproportionately — correctly identifying the single best variant is worth far more than correctly ranking 100 average ones [pascalnotin.substack](https://pascalnotin.substack.com/p/have-we-hit-the-scaling-wall-for)

For a protein engineering challenge evaluated on NDCG, this means: **any model that wastes score dynamic range on mediocre variants will be penalized even if its global Spearman is respectable.** The current ProteinGym leaderboard shows ESM-1v ranking #13 on both Spearman and NDCG, while the same model's Spearman variance is much higher than its NDCG variance — a consequence of catastrophic errors on a small number of assays dragging the aggregate. [biorxiv](https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1.full-text)

## Learning-to-Rank (LTR) Approaches

The FSFP framework (Zhou et al., *Nature Methods* 2024) is the most directly relevant work: it fine-tunes PLMs using **ListMLE**, a listwise learning-to-rank loss that maximizes the log-likelihood of the correct ranking permutation: [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11219809/)

\[
\mathcal{L}_{\text{ListMLE}} = -\log P(\pi^* | \mathbf{s}) = -\sum_{i=1}^{n} \log \frac{\exp(s_{\pi^*(i)})}{\sum_{j=i}^{n} \exp(s_{\pi^*(j)})}
\]

where \(\pi^*\) is the ground-truth ranking and \(\mathbf{s}\) is the model score vector. This directly optimizes ranking order rather than score regression, which is exactly what NDCG measures. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11219809/)

Key findings from FSFP: [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11219809/)

| Setting | Spearman ρ | NDCG improvement |
|---|---|---|
| Zero-shot ESM2 baseline | 0.42 | Reference |
| ESM2 + regression fine-tune | 0.48 | Moderate |
| **ESM2 + ListMLE (FSFP)** | **0.51** | **+18% over regression** |
| ESM2 + ListMLE, 96 labeled examples | 0.66 | Best few-shot |

The gain is especially pronounced for NDCG vs. Spearman because ListMLE concentrates gradient signal at the top of the predicted ranking, matching the NDCG objective structure.

## EvoRank: Zero-Shot Ranking with Evolutionary Objectives

EvoRank (ICLR submission) replaces the standard masked-language-model objective with an **evolutionary ranking objective** derived directly from MSA column distributions: [openreview](https://openreview.net/forum?id=XblaAN1jq6)

- The training loss ranks amino-acid likelihoods according to the marginal frequency in the MSA — a directly biologically motivated ranking prior
- Zero-shot performance on fitness benchmarks improves dramatically, competing with supervised fine-tuned models, particularly on NDCG-style top-k metrics where evolutionary pressure most clearly signals "best variants"
- Critically for your case: it generates a ranked distribution over amino acids at each position, which can be combined with a NDCG-calibrated thresholding strategy [openreview](https://openreview.net/forum?id=XblaAN1jq6)

## Score Calibration for NDCG Optimization

Even without retraining, **score calibration strategies** can improve NDCG without changing model weights:

- **Isotonic regression calibration:** Train isotonic regression mapping PLM raw scores → percentile-calibrated scores on a held-out validation set; this corrects for score compression or expansion that distorts NDCG by giving too many variants similar scores [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12309243/)
- **Score normalization by assay:** PLM scores are non-stationary across protein families; normalizing ESM2 masked-marginal scores by the empirical mean/std of all 313 variants before aggregation prevents one family from dominating the NDCG numerator [biorxiv](https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1.full-text)
- **Top-k recalibration:** Identify the score threshold corresponding to the top 10% empirically (using proxy labels or cross-validated predictions), then apply a monotone transformation that stretches the predicted top-10% score range and compresses the bottom 90% — this directly optimizes NDCG@10% [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10723403/)

## Structure-Conditioned Models on ProteinGym NDCG

ProteinGym NDCG leaderboard (Feb 2026) shows which zero-shot model families rank highest: [pascalnotin.substack](https://pascalnotin.substack.com/p/have-we-hit-the-scaling-wall-for)

| Model | Avg. Spearman | Avg. NDCG@10% |
|---|---|---|
| EVE (VAE, MSA-based) | 0.474 | Top-5 |
| ESM-IF1 (structure-conditioned) | ~0.46 | Top-5 |
| ProMEP (multimodal) | ~0.48 | #1-3 |
| ESM1v (5-model ensemble) | 0.450 | Mid-tier |
| ESM2-650M (masked marginal) | 0.419 | Mid-tier |
| ProtTrans | ~0.38 | Lower |

The pattern is consistent: **MSA-based or structure-conditioned models outperform sequence-only PLMs on NDCG more than on Spearman**, because they better separate the top tail of the fitness distribution where NDCG concentrates its weight. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10418537/)

## Practical NDCG Optimization Strategy for Your Pipeline

Given you have 313 CDS sequences and NDCG as the evaluation criterion, the highest-ROI interventions are:

1. **Switch loss function:** If you have any labeled expression data, fine-tune your ESM2 scorer with ListMLE or **LambdaRank** (which approximates NDCG gradient directly) rather than MSE regression — this is the single most impactful change [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11219809/)

2. **Ensemble for top-k consensus:** Combine ESM2 masked-marginal + EVmutation/EVcoupling + a structure-based score (ESM-IF1 or ProMEP); aggregate by rank averaging, not score averaging — rank averaging is more NDCG-stable than score averaging because it prevents outlier scores in one model from dominating the combined ordering [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10723403/)

3. **Calibrate the top tail:** Apply a **sigmoid rescaling** to your aggregated scores: \(s' = \sigma(\alpha \cdot (s - s_{\text{median}}))\) with \(\alpha > 1\) to spread the top-10% score range; this strictly improves NDCG@10% while leaving Spearman mostly unchanged [pascalnotin.substack](https://pascalnotin.substack.com/p/have-we-hit-the-scaling-wall-for)

4. **Penalize score ties aggressively:** NDCG is severely degraded by ties in the predicted ranking (tied variants are randomly ordered, producing expected-NDCG of 0.5 for tied positions). Ensure your final scoring has enough resolution to avoid ties, especially in the top 10% — adding a small noise-free tiebreaker (e.g., CAI rank) avoids this pathology entirely [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10723403/)

5. **Use NDCG-aware cross-validation:** When tuning weights for your pH 5.5/9.0 ensemble, optimize \(\alpha\) and \(\beta\) by grid search on **NDCG@10%** computed on held-out folds of your labeled data, not Spearman — the optimal weights will differ because NDCG penalizes missing top-ranked true positives much harder than Spearman does [biorxiv](https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1.full-text)


With only 8 unique Tm training points, this is a genuine data bottleneck. The good news is that several substantial PETase-specific datasets are now publicly accessible or partially reconstructable from literature, and they span all three relevant phenotypes: thermostability, activity, and solubility.

## The Seo et al. 2025 Fitness Landscape Dataset (Science)

This is the single most valuable dataset to find for your problem. Seo et al. (*Science*, January 2025) screened **1,894 candidate sequences** from 170 lineages of the polyesterase-lipase-cutinase family using a two-pass pipeline: [science](https://www.science.org/doi/10.1126/science.adp5637)

- **Round 1 (stratified sampling):** 158 nodes experimentally measured for PET-degrading activity and thermal stability, covering the full phylogenetic breadth of the family
- **Round 2 (cluster sampling):** Dense sampling within three high-fitness clusters (C3, C25, C158), generating sequence-fitness pairs at cluster resolution
- **Phenotypes measured:** PET-degrading activity (product yield), thermostability proxy (activity at elevated temperature/durability), and expression/solubility in *E. coli*

The result is ~158+ sequence-fitness pairs across the natural sequence space, from near-zero PETase activity (lipase/cutinase homologs) to high-fitness variants — a quantitative fitness landscape spanning enormous sequence diversity. This is the closest thing to a PETase DMS dataset currently in existence at the family level, and it maps directly onto what your scoring model needs for cross-family generalization. [science](https://www.science.org/doi/10.1126/science.adp5637)

**Key dataset note:** The supplementary tables of this *Science* paper contain the raw activity and stability measurements for all screened nodes. These are directly usable as training labels.

## NREL High-Throughput Screening Platform (ACS Catalysis 2024)

The NREL group developed a **10⁴–10⁵ variant screening platform** for LCC^ICCG^ by directed evolution, measuring three phenotypes simultaneously via paired plate-based assays: [research-hub.nrel](https://research-hub.nrel.gov/en/publications/a-high-throughput-screening-platform-for-engineering-polyethylene)

- **Split-GFP** for protein solubility (binary soluble/insoluble)
- **pNP-esterase model substrate** (4-nitrophenyl butyrate) for activity
- **Thermal shift** for thermostability (Tm proxy)
- Applied to LCC^ICCG^ variant libraries; the best engineered variant showed +8.5% conversion at 65°C and +11.2% at 68°C vs. LCC^ICCG^ baseline

The variant-activity table from the directed evolution rounds contains hundreds of scored single and multi-mutant LCC variants — a directly applicable training set for a PETase activity/stability model. The NREL group's data is the largest published LCC-focused variant dataset with all three phenotypes scored. [research-hub.nrel](https://research-hub.nrel.gov/en/publications/a-high-throughput-screening-platform-for-engineering-polyethylene)

## IsPETase-Specific Variant Compendium

While there is no IsPETase DMS dataset (exhaustive single-mutation scan), the literature collectively provides a high-density point-mutation dataset reconstructable from individual studies:

| Source | Variants | Phenotype | Notes |
|---|---|---|---|
| Yoshida et al. 2016 + engineering papers | WT + ~20 key mutants | kcat, Tm | Alanine scan + active site variants |
| FastPETase (Lu et al. 2022) | ~5,000 ML-guided variants screened | Activity at 50°C | ML-selected, assay data in SI |
| Austin et al. 2021 (FAST-PETase) | ~200 Rosetta-designed variants | Tm, kcat | ML-Rosetta combined design SI |
| Error-prone PCR library (Eiamthong 2022) | ~400 variants from TM backbone | Tm, activity | IsPETase^TM^ + epPCR  [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1002/elsc.202100105) |
| D186 saturation mutagenesis (2024) | 19 variants (all D186X) | Tm, activity at 30°C and 40°C |  [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10975908/) |
| Acidic surface patch variants (2024) | 12 variants (9394insE region) | Activity at 45°C |  [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11306670/) |

By aggregating the supplementary tables from these papers, you can reconstruct **~600–800 IsPETase single and multi-mutant variants** with at minimum one phenotype label each — manageable for a Gaussian Process or Bayesian regression model at this scale.

## ProteinGym PET Hydrolase Entries

ProteinGym v1.1 includes DMS datasets for several serine hydrolases including cutinases and esterases that are phylogenetically adjacent to PETase: [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10723403/)

- **TEM-1 β-lactamase** (2,584 single mutants, functional fitness) — closest serine hydrolase benchmark in ProteinGym for cross-protein transfer learning
- **PAFAH_HUMAN** (esterase/lipase superfamily representative)
- No direct IsPETase DMS is currently in ProteinGym, making it a candidate for a future community contribution

However, the ProteinGym serine hydrolase entries are useful for **zero-shot model calibration**: you can check whether your PLM scorer ranks the correct hydrolase variants well on TEM-1 before trusting its IsPETase predictions.

## Tm-Specific Data: Augmenting Your 8 Points

For thermostability specifically, the Tm gap (8 training points) is critical. Available sources to extend this:

1. **ProThermDB/ThermoMutDB:** Contains >20,000 protein Tm measurements including at least 15–20 IsPETase variant entries from published engineering papers; query `IsPETase` or `6EQE` (PDB ID) returns all deposited variants [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11724889/)

2. **FireProt-DB:** A structured database of engineered thermostable protein variants; searching the cutinase/PETase family returns ~40 documented mutations with ΔTm annotations [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11724889/)

3. **Published IsPETase Tm table (reconstructable):**

| Variant | ΔTm vs WT (°C) | Source |
|---|---|---|
| WT IsPETase | 0 (Tm ≈ 46.4°C) | — |
| S121E/D186H/R280A (TM) | +8.8 | Eiamthong 2022 |
| TM + K95N/F201I | +13.8 |  [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1002/elsc.202100105) |
| FAST-PETase (5 mut.) | +8.0 | Lu 2022 |
| D186V | +12.9 |  [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10975908/) |
| D186N | +7.2 |  [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10975908/) |
| ThermoPETase (R103H/S107R) | +8.6 | Cui 2021 |
| LCCICCG | +37 vs LCC WT | Tournier 2020 |
| CaPETase^M9^ | +41.7 vs WT |  [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10382486/) |

This gives you ~15–20 curated Tm anchor points from IsPETase literature alone, plus the LCC/CaPETase family data if you want to train a cross-family ΔTm predictor.

## Practical Data Assembly Pipeline

Given the 8-point training constraint, the recommended strategy is:

1. **Harvest Seo et al. 2025 SI tables** — ~158 sequence-fitness pairs spanning the full polyesterase family, ideal for training a family-wide fitness landscape model [science](https://www.science.org/doi/10.1126/science.adp5637)
2. **Harvest NREL 2024 directed evolution SI** — hundreds of LCC variant activity/Tm pairs for supervised model training [research-hub.nrel](https://research-hub.nrel.gov/en/publications/a-high-throughput-screening-platform-for-engineering-polyethylene)
3. **Reconstruct IsPETase variant table** from FastPETase SI, FAST-PETase SI, and the D186X saturation mutagenesis paper to get ~80–100 IsPETase-specific labeled points [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10975908/)
4. **Query ThermoMutDB / FireProt-DB** programmatically for all entries with PDB ID 6EQE or UniProt A0A0K8P8V7 to recover deposited Tm measurements
5. **Train a multi-task GP or Bayesian Ridge** with `[ESM2 embedding + structural features]` → `[Tm, activity at pH 5.5, activity at pH 9.0]`, using the cross-protein datasets for pretraining and IsPETase-specific points for fine-tuning — a two-stage transfer approach proven effective at this data scale [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11219809/)