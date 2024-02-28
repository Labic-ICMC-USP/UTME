# UTME for Climate Change Analsys

To create strcutures holding large relevant data regarding climate change events we need to be effectively able to extract this infromation from sources such as news, and that is the objective of this class. This tutorial explores the motivation behind using UTME and the importance of automated mining and monitoring of climate change texts, it is extremely based on the tutorial for hate speech analysis which can be found [here](https://github.com/Labic-ICMC-USP/UTME/tree/main/tutoriais/hatespeech).

### Motivation for Using UTME

Traditional methods of text classification often face limitations in adaptability across diverse taxonomies due to the requirement for extensive labeled datasets. UTME provides a versatile and unsupervised approach, allowing users to classify documents within a custom hierarchical taxonomy without the need for labeled data. The unsupervised taxonomy expansion feature further empowers users to dynamically generate subcategories based on document content, offering a granular understanding of the data.

### Step-by-Step Tutorial: Putting UTME into Action

Now, let's walk through a step-by-step on how to leverage UTME for climate change detection:

1. **Initialize UTME:**
   - Set up your UTME environment. UTME leverages Large Language Models (LLMs), demonstrating its effectiveness across proprietary, open-source, and low-computational-cost models. Its adaptability makes it a practical choice for users and organizations seeking advanced text mining capabilities without requiring extensive computational resources.


```python
from utme.UTME import UTME
from utme.BinaryClassifier import BinaryClassifier
from utme.TaxonomyClassifier import TaxonomyClassifier
from utme.SubcategoryGenerator import SubcategoryGenerator

# Initialize UTME with LLM (Language Model) credentials
llm_key = "your_llm_key"
llm_endpoint = "your_llm_url_endpoint"
llm_options = {'model': "openchat_3.5", 'max_tokens': 1024}
utme_base = UTME(llm_endpoint, llm_key, llm_options)
```


2. **Define Context and Taxonomy:**
   - Clearly define the context and taxonomy relevant to climate change detection. This step establishes the framework for classification.


```python
# Define context and taxonomy for climate change analysis
context = '''In the context of climate change, a myriad of events and issues coalesce to shape the environmental landscape. Extreme weather events, such as hurricanes and droughts, intertwine with concerns about rising sea levels and melting ice caps. Ecosystem changes and biodiversity preservation efforts are juxtaposed against the challenges of deforestation and the impact on marine life. Greenhouse gas emissions and the ensuing air quality deterioration pose health risks, necessitating a focus on renewable energy solutions and sustainable agricultural practices. The urban heat island effect and corporate sustainability challenges contribute to broader issues of climate-related migration patterns and water scarcity. Advocacy for climate action, policy formulation, and technological innovations grapple with challenges like waste management, ocean conservation, and circular economy promotion. Artistic expression, cultural adaptations, and climate-responsible tourism underscore the cultural dimensions of climate change, while scientific collaboration, climate justice advocacy, and smart city initiatives strive to address these complex issues. The interconnected nature of these challenges emphasizes the need for global cooperation, public-private partnerships, and ethical consumer choices to navigate the complexities of climate change and build a resilient, sustainable future.'''

taxonomy = '''0 NONE
1 Extreme Weather Events
2 Ecosystem Changes
3 Greenhouse Gas Emissions
4 Health Risks
5 Water Scarcity Issues
6 Melting Ice Caps
7 Urban Heat Island Effect
8 Deforestation Impacts
9 Air Quality Deterioration
10 Biodiversity Preservation
11 Sustainable Agriculture Challenges
12 Impact on Marine Life
13 Rising Sea Levels
14 Carbon Footprint Reduction
15 Agricultural Adaptation Challenges
16 Community Resilience Issues
17 Renewable Transportation Challenges
18 Green Building Technology Issues
19 Corporate Sustainability Challenges
20 Climate-Related Migration Patterns
21 Water Resource Management Issues
22 Renewable Energy Accessibility
23 Waste Management Challenges
24 Ocean Conservation Challenges
25 Policy Advocacy for Climate Action
26 Technology Transfer for Climate Mitigation
27 Green Innovation Challenges in Industries
28 Cultural Adaptations to Climate Change
29 Promotion of Circular Economy Challenges
30 Artistic Expression on Climate Issues
31 Climate-Responsible Tourism Challenges
32 Scientific Collaboration on Climate Studies
33 Climate Justice Advocacy
34 Smart Cities for Climate Resilience
35 Climate-Responsive Urban Planning
36 Public-Private Partnerships in Climate Action
37 Renewable Energy Accessibility Issues
38 Youth Engagement in Climate Initiatives
39 Ethical Consumer Choices for Climate Impact
40 Displacement and Climate Refugees
41 Political Challenges in Climate Agreements
42 Climate Change Impact on Indigenous Peoples
43 Conflict Over Natural Resources
44 Food Security Challenges
45 Threats to Global Trade
46 Challenges in Climate Diplomacy
47 Climate-Induced Economic Inequalities
48 Technological Gaps in Climate Solutions
49 Global Cooperation Challenges in Climate
50 Challenges in Addressing Climate Denial'''
```

In the UTME framework, the definition of context and taxonomy is flexible, requiring a degree of experimentation and creativity. It can be tailored to the specifics of the Large Language Model (LLM) used by UTME. While context is crucial throughout the process, it plays a pivotal role in BinaryClassification, thereby filtering texts of interest before taxonomy mapping.

For the taxonomy, used in TaxonomyClassifier, there is a specific format to follow. Each taxonomy item is a line starting with an increasing numerical identifier. It is imperative that the initial taxonomy item is "0 NONE", serving as the second filter for texts of interest.

3. **Binary Classification for Text Filtering:**
   - Use the Binary Classifier to filter texts of interest, focusing on documents that may contain climate change.

```python
# Start BinaryClassifier to filter documents of interest
bc = BinaryClassifier(utme_base, context)
y_pred = bc.classify(df.text.to_list())
df['y_pred'] = y_pred
df_filtered = df[df.y_pred == 'YES']
```

The BinaryClassifier module in UTME employs a few-shot prompt learning approach, utilizing only the provided context to make predictions. It operates by prompting the model with the contextual information and expects a binary response: YES for climate change events and NO for non related to climate change events. 

4. **Taxonomy Mapping – First Level:**
   - Employ the Taxonomy Classifier to map documents within the predefined hierarchical taxonomy at the first level.

```python
# Start TaxonomyClassifier to map documents to predefined categories (First Level)
tc = TaxonomyClassifier(bc, taxonomy)
taxonomy_pred = tc.classify(df_filtered.text.to_list())
df_filtered['level1'] = taxonomy_pred
df_filtered_level1 = df_filtered[~df_filtered.level1.str.contains('NONE')]
```

The TaxonomyClassifier utilizes a user-defined taxonomy tailored for climate change events detection. The predefined taxonomy consists of distinct levels, such as Greenhouse Gas Emissions, Deforestation Impacts, Rising Sea Levels, Extreme Weather Events, and more. By employing this module, users can effectively categorize and identify climate change events within their text data, contributing to a comprehensive understanding of the diverse themes associated to the variations in climate.

It's worth noting that for the TaxonomyClassifier, only the structure of the climate change taxonomy is needed, without relying on labeled data for training. 

5. **Unsupervised Taxonomy Expansion – Second Level:**
   - Utilize the Subcategory Generator to dynamically expand the taxonomy, generating subcategories for more detailed analysis.

```python
# Perform Unsupervised Taxonomy Expansion (Second Level)
L = []
for category in df_filtered_level1.level1.unique():
    sg = SubcategoryGenerator(tc)
    df_category = df_filtered_level1[df_filtered_level1.level1 == category]

    # Generate subcategories for the selected category
    subcategories = sg.generate_subcategories(category, df_category.sample(expansion_sample_size, replace=True).text.to_list())

    # Classify documents using the expanded subcategories
    tc2 = TaxonomyClassifier(bc, subcategories)
    taxonomy_pred2 = tc2.classify(df_category.text.to_list())

    df_category['level2'] = taxonomy_pred2
    L.append(df_category)
df_filtered_level2 = pd.concat(L)
```

The SubcategoryGenerator in UTME is essential for unsupervised taxonomy expansion, creating subcategories within predefined categories. For example, if the TaxonomyClassifier identifies a document under "Greenhouse Gas Emissions," the SubcategoryGenerator can generate subcategories like "Carbon Emissions Impacts" or "Rising Methane Emissions" based on the document content. This unsupervised approach enables the system to discover subtopics without labeled data, enhancing exploratory analysis and taxonomy development.

Access df_filtered_level2 to see the mapping result for each document.

6. **Graph-Based Analysis for Exploration:**
   - Leverage UTME's graph-based analysis to visually explore document relationships, aiding in the identification and monitoring of climate change patterns.


```python
# Start SubcategoryGenerator for graph generation
sg = SubcategoryGenerator(tc)
sg.graph_generation(df_filtered_level2)
sg.graph_export_cosmograph(df_filtered_level2)
``` 
![Graph - Climate Change Analysis](https://raw.githubusercontent.com/Labic-ICMC-USP/UTME/main/climateChange/UTME2.png "Graph - Climate Change Analysis")

The UTME also facilitates graph analysis through the generated nodes.csv and edges.csv files, allowing for exploratory analysis of the results. In this graph, each climate change text serves as a vertex, and similar texts are connected. UTME generates connections by exploring both document similarity and the predefined climate change taxonomy. For analyzing large graphs, the Cosmograph app is recommended, providing robust features for graph visualization and exploration.

To navigate the taxonomy, we also suggest using TreeMaps:

```python
# Treemap
import plotly.express as px
df_tree = utme_climatechange.df_filtered_level2[['text','level1','level2']]
fig = px.treemap(df_tree, path=['level1', 'level2'],  color='level2',  color_continuous_scale='RdBu')
fig.show()
``` 
![Graph - Climate Change Analysis](https://raw.githubusercontent.com/Labic-ICMC-USP/UTME/main/climateChange/UTME1.png "TreeMap - Climate Change Analysis")


# UTME_ClimateChange Class

The UTME_ClimateChange class consolidates all the stages of UTME into a single, user-friendly interface, simplifying the climate change analysis process for users and analysts. This class encapsulates the Binary Classifier for text filtering, the Taxonomy Classifier for mapping documents to a predefined taxonomy, and the Unsupervised Taxonomy Expansion through the Subcategory Generator.

```python
# UTME Climate Change Analysis Tutorial

from utme.UTME import UTME
from utme.BinaryClassifier import BinaryClassifier
from utme.TaxonomyClassifier import TaxonomyClassifier
from utme.SubcategoryGenerator import SubcategoryGenerator
import pandas as pd

class UTME_ClimateChange():

    def __init__(self, utme_base):
        self.utme_base = utme_base

    def start(self, df, context, taxonomy, expansion_sample_size=10):
        """
        Start the UTME_ClimateChange process for classifying climate change in text data.

        Parameters:
        - df (pd.DataFrame): DataFrame containing text data to be processed.
        - context (str): Example of domain-application interest documents.
        - taxonomy (list): List of predefined categories for taxonomy classification.
        - expansion_sample_size (int): Size of the sample for taxonomy expansion (default is 10).
        """
        print('# Binary Classifier for Text Filtering')
        bc = BinaryClassifier(self.utme_base, context)
        y_pred = bc.classify(df.text.to_list())
        df['y_pred'] = y_pred
        df_filtered = df[df.y_pred == 'YES']
        print('Selected', len(df_filtered), 'documents of', len(df))
        self.df_filtered = df_filtered

        print('# Taxonomy Document Mapping - First Level')
        tc = TaxonomyClassifier(bc, taxonomy)
        taxonomy_pred = tc.classify(df_filtered.text.to_list())
        df_filtered['level1'] = taxonomy_pred
        df_filtered_level1 = df_filtered[~df_filtered.level1.str.contains('NONE')]
        print('Selected', len(df_filtered_level1), 'documents of', len(df_filtered))
        self.df_filtered_level1 = df_filtered_level1

        print('# Unsupervised Taxonomy Expansion - Second Level')
        L = []
        for category in df_filtered_level1.level1.unique():
            print('Expanding:', category)
            sg = SubcategoryGenerator(tc)
            df_category = df_filtered_level1[df_filtered_level1.level1 == category]


            # Generate subcategories for the selected category
            sample_list = df_category.sample(expansion_sample_size, replace=True).text.to_list()
            subcategories = sg.generate_subcategories(category, sample_list)

            # Classify documents using the expanded subcategories
            tc2 = TaxonomyClassifier(bc, subcategories)
            taxonomy_pred2 = tc2.classify(df_category.text.to_list())

            df_category['level2'] = taxonomy_pred2
            L.append(df_category)
        df_filtered_level2 = pd.concat(L)
        self.df_filtered_level2 = df_filtered_level2

        print('# Graph generation for exploratory analysis')
        sg = SubcategoryGenerator(tc)
        sg.graph_generation(df_filtered_level2)
        sg.graph_export_cosmograph(df_filtered_level2)

        print('# UTME completed')

# Usage Example
llm_key = "your_llm_key"
llm_endpoint = "your_llm_url_endpoint"
llm_options = {'model': "openchat_3.5", 'max_tokens': 1024}
utme_base = UTME(llm_endpoint, llm_key, llm_options)
utme_climatechange = UTME_ClimateChange(utme_base)


# Dataset

# This Dataset was taken from the site (), you need to download it (or another dataser) to run this code. For this dataset in specific its texts are on a column named Article so you need to change the name of the column to text so that the code works.

df_climatechange = pd.read_csv('global-issues.csv')

df = df_climatechange.sample(500)

df.rename(columns = {'Article':'text'}, inplace = True)

df  # important: remove very short texts

# Define context and taxonomy for climate analysis
context = '''In the context of climate change, a myriad of events and issues coalesce to shape the environmental landscape. Extreme weather events, such as hurricanes and droughts, intertwine with concerns about rising sea levels and melting ice caps. Ecosystem changes and biodiversity preservation efforts are juxtaposed against the challenges of deforestation and the impact on marine life. Greenhouse gas emissions and the ensuing air quality deterioration pose health risks, necessitating a focus on renewable energy solutions and sustainable agricultural practices. The urban heat island effect and corporate sustainability challenges contribute to broader issues of climate-related migration patterns and water scarcity. Advocacy for climate action, policy formulation, and technological innovations grapple with challenges like waste management, ocean conservation, and circular economy promotion. Artistic expression, cultural adaptations, and climate-responsible tourism underscore the cultural dimensions of climate change, while scientific collaboration, climate justice advocacy, and smart city initiatives strive to address these complex issues. The interconnected nature of these challenges emphasizes the need for global cooperation, public-private partnerships, and ethical consumer choices to navigate the complexities of climate change and build a resilient, sustainable future.'''

taxonomy = '''0 NONE
1 Extreme Weather Events
2 Ecosystem Changes
3 Greenhouse Gas Emissions
4 Health Risks
5 Water Scarcity Issues
6 Melting Ice Caps
7 Urban Heat Island Effect
8 Deforestation Impacts
9 Air Quality Deterioration
10 Biodiversity Preservation
11 Sustainable Agriculture Challenges
12 Impact on Marine Life
13 Rising Sea Levels
14 Carbon Footprint Reduction
15 Agricultural Adaptation Challenges
16 Community Resilience Issues
17 Renewable Transportation Challenges
18 Green Building Technology Issues
19 Corporate Sustainability Challenges
20 Climate-Related Migration Patterns
21 Water Resource Management Issues
22 Renewable Energy Accessibility
23 Waste Management Challenges
24 Ocean Conservation Challenges
25 Policy Advocacy for Climate Action
26 Technology Transfer for Climate Mitigation
27 Green Innovation Challenges in Industries
28 Cultural Adaptations to Climate Change
29 Promotion of Circular Economy Challenges
30 Artistic Expression on Climate Issues
31 Climate-Responsible Tourism Challenges
32 Scientific Collaboration on Climate Studies
33 Climate Justice Advocacy
34 Smart Cities for Climate Resilience
35 Climate-Responsive Urban Planning
36 Public-Private Partnerships in Climate Action
37 Renewable Energy Accessibility Issues
38 Youth Engagement in Climate Initiatives
39 Ethical Consumer Choices for Climate Impact
40 Displacement and Climate Refugees
41 Political Challenges in Climate Agreements
42 Climate Change Impact on Indigenous Peoples
43 Conflict Over Natural Resources
44 Food Security Challenges
45 Threats to Global Trade
46 Challenges in Climate Diplomacy
47 Climate-Induced Economic Inequalities
48 Technological Gaps in Climate Solutions
49 Global Cooperation Challenges in Climate
50 Challenges in Addressing Climate Denial'''

utme_climatechange.start(df, context, taxonomy, 3)
```


## UTME ClimateChange in Action using Google Colab

### Google Colab 1: LLM Server

To get a hands-on experience with the UTME ClimateChange functionality and test its concepts, you can access the  Google Colab notebook [here](https://colab.research.google.com/drive/1mvLCbjmTFU_Fn3YLgGarLEhN_pks2yNj?usp=sharing). This notebook provides a simple Large Language Model (LLM) server setup to explore the basic capabilities of the UTME, allowing you to understand the Binary Classifier, Taxonomy Classifier, and Subcategory Generator. This serves as a preliminary step to familiarize yourself with the UTME ClimateChange workflow.

### Google Colab 2: UTME ClimateChange Implementation and Experiment

For a more comprehensive demonstration, the second Google Colab notebook [here](https://colab.research.google.com/drive/1XMblXvW71MQHMXhhOuFrvqIzBvP08mqU?usp=sharing) showcases the full implementation of the UTME_ClimateChange class. Additionally, a small experiment using climate change texts available in the [Abdul Raheem Aleem Global Issues dataset](https://www.kaggle.com/datasets/abdulraheemaleem/globalissues) is included. This notebook shows you an application of the UTME ClimateChange class on real climate change data, demonstrating the effectiveness of the solution in filtering, classifying, and expanding the taxonomy for a more detailed climate change analysis.

Feel free to explore these notebooks to gain practical insights into the UTME ClimateChange capabilities and witness its application in addressing climate change challenges within textual datasets.
