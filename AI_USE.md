# AI use and reflection

This statement accompanies my MMA3001 report on detecting annotated unsealed pork-packaging regions. It explains the assistance provided by ChatGPT/Codex, my contribution and the verification evidence.



### Tools and purpose

I used ChatGPT/Codex to generate the initial Python/Google Colab code, troubleshoot errors and improve the report’s wording and structure. I sought this assistance because I lacked confidence in coding and academic writing.



AI also assisted with project planning, software-verification code, documentation, metric explanations, model-comparison recommendations and prediction-image observations. It provided an executable starting point and step-by-step support with the Colab and GitHub workflow.



The model metrics, timings, checkpoints and prediction plots came from my executed Colab experiments. AI helped explain and present these results.Approximate contribution

Estimates refer to AI's contribution to preparing and interpreting the work,
not to the fraction of training performed by AI. Exact line or word counts are
not required for an honest approximate estimate.


Approximate contribution
---

|Area|Approximate AI contribution|My contribution|
|-|-|-|
|Code|High: AI provided most of the initial notebook and verification code and helped troubleshoot errors.|Running the notebooks, applying suggested corrections and saving outputs.|
|Writing|High: AI drafted substantial parts of the documentation and report and improved their wording and structure.|Providing project information and organising the submission materials.|
|Analysis|Substantial: AI explained metrics, recommended model and resolution choices and helped draft image observations.|Executing the experiments and preserving the measured results used in the analysis.|

## My work and decision responsibility

I ran the Colab experiments and worked through the repository upload. AI recommended the single-class unsealed target, retention of target-negative images, matched 30-epoch YOLOv8n/YOLOv8s runs and the resolution study.



I followed these recommendations and executed the experiments. The recorded validation results support selecting YOLOv8n at 640 pixels. These choices were AI-assisted; I did not independently develop all the settings or experimental methods.



My contribution included carrying out the workflow and preserving its results. I remain responsible for reviewing the recommendations, submitted claims and engineering interpretation.

Verification of code, results and observations
---

The verification workflow included:



* A saved Colab software-test report recording 17 passing known-input tests on 6 October 2026. These cover preparation and entry-point behaviours, rather than proving detector accuracy.



* An abstract-syntax-tree comparison checking that the tested preparation function matched the function in the saved training notebook.



* An actual-archive audit recording the original ZIP hash and checking prepared image and target-box counts for each split.



* Report tables derived from the saved validation comparison, resolution study and final-test JSON. CPU test timing was kept separate from T4 validation timing.



* AI-assisted comparison of saved annotation and prediction images. Observations were restricted to visible detections and box alignment in the displayed eight-image batch. The full-test confusion matrix was considered separately.



The model and resolution recommendations used validation evidence. The final test data were not used to change the selected design.



Verification relied on automated tests and AI-assisted interpretation. These activities should therefore be distinguished from independent manual review. The original scaffold checks, stored Colab test report and fresh-clone verification are separate records.Errors and their implications


Errors and their implications
---

|Issue encountered|Correction or implication|
|-|-|
|`TRUE` caused a Python `NameError`|Python's Boolean literal is `True`.|
|`ultralytics` was missing|Install the recorded package in the active runtime before importing it.|
|`RUN\\\_DIR` and other variables were unavailable|Re-establish configuration in execution order after a runtime reset and select the intended saved run.|
|CPU evaluation was slow|Record the actual device and keep CPU and GPU timings separate.|
|An upload flattened repository folders|Preserve the expected `notebooks/`, `tests/`, `docs/`, `results/` and `verification/` paths and verify a fresh clone.|



I used AI guidance to work through these problems. They highlighted the importance of correct syntax, package installation, cell execution order and folder structure. They also show why generated code and instructions require execution and inspection. Their occurrence does not establish that AI caused every error.


## Personal learning and remaining limitations

**AI assistance helped me progress through unfamiliar coding and documentation tasks. The workflow highlighted the distinction between checking software behaviour and evaluating model performance. Further independent review is still needed to strengthen my understanding of the code, metrics and engineering interpretation.**



**The software audit does not establish physically correct annotations or independent dataset splits. Related conveyor frames may overlap across splits. The test contains only 80 images and 49 target boxes, only one training seed was studied, and missed regions and imperfect localisation remain.**



**The detector predicts the supplied visual annotation and does not certify airtightness, meat freshness or food safety. I remain responsible for all submitted code, results, references and engineering decisions.**

