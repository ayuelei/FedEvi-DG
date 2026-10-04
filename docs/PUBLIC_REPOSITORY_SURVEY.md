# What comparable paper repositories provide

Reviewed on 2026-10-04. This is an organizational survey of official repository READMEs and file listings, not an audit of experimental correctness. Linked material may change after this date. No external repository code or datasets have been vendored here.

| Repository | Observed release contents | Practice used here |
|---|---|---|
| [FedDG-ELCFS](https://github.com/liuquande/FedDG-ELCFS) | Network and loader modules, data preparation, training/testing, frequency interpolation demo, data layout, citation and upstream acknowledgements | Separate preparation, training, evaluation and a small demo |
| [FACT](https://github.com/MediaBrain-SJTU/FACT) | Model/data modules, train/test launchers, environment versions, external dataset instructions, train/validation lists, citation and upstream acknowledgements | Document environment, source/target domains and fixed split protocol |
| [CCST](https://github.com/JeremyCJM/CCST) | Requirements, dataset links, style-transfer stages, federated training, testing, model-download instructions, additional run commands, license | Document the ordered workflow and keep data acquisition external |
| [HarmoFL](https://github.com/med-air/HarmoFL) | Environment YAML, model/federated/data utilities, train/test shell entry points and setup instructions | Provide dependencies and executable experiment entry points |
| [FedBN](https://github.com/med-air/FedBN) | Environment and requirements, federated implementations, data and pretrained-model download instructions, training/testing and citation | Separate code, datasets and optional checkpoint distribution |
| [FSCL-BC](https://github.com/faisalahm3d/FSCL-BC) | Client/server, ResNet18 model, contrastive loss, data utilities, hyperparameters and entry point; label-file format and output-metric descriptions | Make the final method, inputs and outputs unambiguous |

These repositories do not all provide the same level of packaging. Some ask users to edit paths; some offer external data or weight downloads. Their presence is not a reason to redistribute breast ultrasound images or copy their evaluation choices. In particular, FedEvi-DG keeps source-only checkpoint selection, normalization, temperature fitting and deployment-threshold selection.

FedEvi-DG additionally supplies synthetic tests, bootstrap analysis, patient-aggregation sensitivity, explanation interventions, aggregate reference tables, and data-free source/cleanup validation records. The current implementation is limited to the paper's methods and controls. The survey does not assert that any reference repository lacks an item not visible in its inspected README/listing.
