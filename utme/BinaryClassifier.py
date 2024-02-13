from tqdm import tqdm

class BinaryClassifier():
    def __init__(self, utme, context, attempts=3):
        """
        BinaryClassifier class for performing binary classification using the UTME model.
        
        Parameters:
        - utme (UTME): UTME model instance.
        - context (str): Example of domain-application interest documents.
        - attempts (int): Number of attempts to retry classification in case of an error (default is 3).
        """
        self.utme = utme
        self.context = context
        self.attempts = attempts

    def classify(self, documents, show_progress=True):
        """
        Classify a list of documents into binary categories (e.g., YES/NO).
        If an error occurs during classification, it returns a special label called ERROR.

        Parameters:
        - documents (list): List of documents for classification.
        - show_progress (bool): Flag to indicate whether to display progress bar (default is True).

        Returns:
        - y_pred (list): List of binary classifications for each document.
        """
        y_pred = []

        # Iterate through each document for classification
        for document in tqdm(documents, desc="Classifying Documents", disable=not show_progress):

            try:
                # Construct the prompt for binary classification
                prompt = '@CONTEXT = ' + self.context + '\n#####\n'
                prompt += '@DOCUMENT = ' + document + '\n#####\n'
                prompt += '''Does the @DOCUMENT belong to the @CONTEXT? Just answer with YES or NO.'''

                # Make an attempt to classify the document
                response = self.utme.llm_prompt(prompt)
                s = response['choices'][0]['message']['content'].strip().upper()

                # Retry classification if the response is not YES or NO
                for _ in range(self.attempts):
                    if s in ['YES', 'NO']:
                        break
                    response = self.utme.llm_prompt(prompt)
                    s = response['choices'][0]['message']['content'].strip().upper()

                # If still not classified as YES or NO, label as ERROR
                if s not in ['YES', 'NO']:
                    s = 'ERROR'
                
                # Append the classification result to the list
                y_pred.append(s)
                
            except Exception as error:
                # Handle exceptions during classification and label as ERROR
                y_pred.append('ERROR')
                print("An exception occurred:", error)

        return y_pred
