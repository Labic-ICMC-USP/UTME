from sentence_transformers import SentenceTransformer
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import pandas as pd

class SubcategoryGenerator():
    def __init__(self, taxonomy_classifier, attempts=7):
        """
        SubcategoryGenerator class for generating subcategories within predefined categories.
        
        Parameters:
        - taxonomy_classifier: TaxonomyClassifier instance for generating subcategories.
        - attempts (int): Number of attempts to retry generation in case of an error (default is 7).
        """
        self.taxonomy_classifier = taxonomy_classifier
        self.attempts = attempts
        self.utme = taxonomy_classifier.utme
        self.text_encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    def generate_subcategories(self, item_taxonomy, document_sample, max_subtopics=7):
        """
        Generate subcategories for a given document sample.
        
        Parameters:
        - item_taxonomy (str): The main taxonomy category for subcategory generation.
        - document_sample (list): List of documents for subcategory generation.
        - max_subtopics (int): Maximum number of subtopics to generate (default is 7).

        Returns:
        - subcategories (str): Generated subcategories for the specified item_taxonomy.
        """
        documents = '\n'.join(document_sample)

        for attempt in range(self.attempts):
            try:
                prompt = '@CONTEXT = ' + self.taxonomy_classifier.binary_classifier.context + '\n#####\n'
                prompt += '@DOCUMENTS = ' + documents + '\n#####\n'
                prompt += f'''Generate, list, and enumerate top-{max_subtopics} specific subtopics for general topic "{item_taxonomy}" from @DOCUMENTS content:\n\n'''

                response = self.utme.llm_prompt(prompt)

                s = response['choices'][0]['message']['content'].strip().split("\n")
                checking = sum(str(num) in s[num-1] for num in range(1, len(s)+1))

                if checking == len(s):
                    return response['choices'][0]['message']['content'].strip()

            except Exception as error:
                print("An exception occurred:", error)

        return None

    def graph_generation(self, df_classified_documents):
        """
        Generate a graph based on document embeddings and connectivity.

        Parameters:
        - df_classified_documents (pd.DataFrame): DataFrame (text,level1,level2) containing classified documents.
        """
        text_encoder = self.text_encoder
        embeddings = text_encoder.encode(df_classified_documents.text.to_list())

        A = kneighbors_graph(embeddings, 1, mode='connectivity', include_self=False)
        G = nx.Graph(A.toarray())

        mapping = {counter: index for counter, index in enumerate(df_classified_documents.index)}
        G = nx.relabel_nodes(G, mapping)

        for category in df_classified_documents.level1.unique():
            print(category)
            df_temp = df_classified_documents[df_classified_documents.level1 == category]
            embeddings2 = text_encoder.encode(df_temp.text.to_list())
            
            if embeddings2.shape[0] >= 5:
                A2 = kneighbors_graph(embeddings2, 3, mode='connectivity', include_self=False)
                G2 = nx.Graph(A2.toarray())
                mapping = {counter: index for counter, index in enumerate(df_temp.index)}
                G2 = nx.relabel_nodes(G2, mapping)
                G.add_edges_from(G2.edges())

        self.G = G
        L_edges = [[edge[0], edge[1], 1] for edge in G.edges()]
        df_edges = pd.DataFrame(L_edges, columns=['source', 'target', 'value'])
        self.df_edges = df_edges
        print('Graph generation: OK')

    def graph_export_cosmograph(self, df_classified_documents):
        """
        Export nodes and edges data for CosmoGraph visualization.

        Parameters:
        - df_classified_documents (pd.DataFrame): DataFrame containing classified documents.
        """
        print('Saving: nodes.csv')
        df_classified_documents[['text', 'level1', 'level2']].to_csv('nodes.csv', sep=',', index_label='id')
        print('Saving: edges.csv')
        self.df_edges.to_csv('edges.csv', sep=',')
        print('Hint: Open https://cosmograph.app/ and use edges.csv and nodes.csv to visualize the graph.')

    def sample_from_graph(self, graph, category, sampling_method='random'):
        """
        Sample documents from the graph for a specific category using a specified sampling method.
        
        Parameters:
        - graph (nx.Graph): The graph containing documents.
        - category (str): The category for which documents are sampled.
        - sampling_method (str): The method used for sampling (default: 'random').
        """
        # Implement logic to sample documents from the graph for a specific category
        pass  # Placeholder for the implementation
