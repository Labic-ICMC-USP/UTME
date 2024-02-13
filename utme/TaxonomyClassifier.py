from tqdm import tqdm

class TaxonomyClassifier():
    def __init__(self, binary_classifier, taxonomy, attempts=5):
        """
        TaxonomyClassifier class for performing taxonomy-based classification using the UTME model.
        
        Parameters:
        - binary_classifier: Model for performing binary classification using the UTME model.
        - taxonomy (list): List of predefined categories for taxonomy classification.
        - attempts (int): Number of attempts to retry classification in case of an error (default is 5).
        """
        self.binary_classifier = binary_classifier
        self.utme = binary_classifier.utme
        self.taxonomy = taxonomy
        self.attempts = attempts

    def classify(self, documents):
        """
        Classify a list of documents into predefined categories based on taxonomy.

        Parameters:
        - documents (list): List of documents for classification.

        Returns:
        - y_pred (list): List of predicted categories for each document.
        """
        y_pred = []

        # Iterate through each document for taxonomy-based classification
        for document in tqdm(documents, desc="Classifying Documents"):

            try:
                # Construct the prompt for taxonomy-based classification
                prompt = '@CONTEXT = ' + self.binary_classifier.context + '\n#####\n'
                prompt += '@DOCUMENT = ' + document + '\n#####\n'
                prompt += '''Select the @TAXONOMY items that best classify the @DOCUMENT according to its textual content. The output should only be items from the @TAXONOMY (one per line).'''
                prompt += '\n\n@TAXONOMY:\n'
                prompt += self.taxonomy

                # Attempt to classify the document into a taxonomy category
                category = '0 NONE'
                for _ in range(self.attempts):
                    try:
                        response = self.utme.llm_prompt(prompt)
                        s = response['choices'][0]['message']['content'].strip().upper().split("\n")
                        
                        # Extract the category from the response
                        for item in s:
                            if len(item) > 2 and item[0].isnumeric() and item in self.taxonomy.upper():
                                category = item
                                break
                    except:
                        pass

                y_pred.append(category)

            except Exception as error:
                # Handle exceptions during classification and label as '0 NONE'
                y_pred.append('0 NONE')
                print("An exception occurred:", error)

        return y_pred
