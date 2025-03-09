# evaluate_chatbot_llms_for_healthcare

The rapid adoption of large language models (LLMs) through chatbots has opened up new possibilities in healthcare, offering AI-driven support. Despite their potential, the literature presents mixed reviews on the effectiveness of LLMs in healthcare—highlighting both promising capabilities and challenges in certain applications.

So we conducted a study using MIMIC-IV data  systematically compares the performance of  the LLMs in critical healthcare tasks. Specifically, we assess their ability to generate primary diagnoses, map diagnoses to ICD-9 codes, and predict risk stratification for hospital readmissions. We also evaluate the models’ reasoning capabilities, focusing on their ability to explain the rationale behind their diagnoses and readmission predictions.

In this repository the following scripts can be found
1. **mimic_sampling_and_prompt_creation.py**- This py file shows how the sample of 300 subject_ids were created- 150 subjects with >1 admission and the rest 150 have no redamission. Once sampling is complete, the key sections of structured clinical note s are extraced abd they are tsringed together programatically to create prompt
2. **extraction_from_llm.py**- This py file contains script to help extract the response of LLMs into a structured CSV. Each response from each prompt is collected but it initially exists as a paragraph. to make this usable for analaysis, we have to extract them into a structured format in CSV.
3. **analysis_and_visualization.py** - This py file has script that helps conduct comparitive analysis of top 10 icd codes from mimic-iv sample, llama, chat-gpt, gemini. It has has comparitive analysis of the icd-9 codes generated by each LLM vs MIMIC-IV sample, redamission risk comparison, primary diagnosis comparison and F1-score for icd-9 code prediction and redamission risk prediction.

MIMIC-IV data files are not provided here as they are restricted access file thta cannot be made accessed publically. Once you have access to those files, these scripts can be used to re-create the analysis.

By comparing the performance of these models in healthcare contexts, this study provides valuable insights into their strengths, limitations, and potential for future use in clinical decision-making
