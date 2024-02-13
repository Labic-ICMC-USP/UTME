# UTME: Unsupervised Taxonomy Mapping and Expansion for Document Classification

Document classification within a custom internal hierarchical taxonomy is a prevalent challenge for organizations dealing with textual data. Traditional approaches rely on supervised techniques, effective on specific datasets but constrained by the need for extensive corpora of annotated documents. Furthermore, these models lack direct applicability to different taxonomies. In this repository, we contribute to this issue by introducing a methodology for classifying text within a custom hierarchical taxonomy entirely in the absence of labeled data. Our approach incorporates unsupervised taxonomy mapping for first-level document classification, taxonomy and unsupervised taxonomy expansion for dynamic adaptation to evolving content.

## Key Features

1. **Unsupervised Taxonomy Mapping:**
   - Classifies documents within a custom user-defined hierarchical taxonomy without the need for labeled data.

2. **Unsupervised Taxonomy Expansion:**
   - Expands and adapts taxonomies in an unsupervised manner.
   - Utilizes document content to identify and generate new subcategories dynamically.

4. **Graph-Based Document Relationships:**
   - Constructs a graph based on document similarity for exploratory visual analysis and data sampling.

## Getting Started

Read our tutorials to see examples of the UTME in action:

* [UTME for Hate Speech Analsys](https://github.com/Labic-ICMC-USP/UTME/tree/main/tutoriais/hatespeech): This tutorial explores the motivation behind using UTME and the importance of automated mining and monitoring of hate speech texts.
