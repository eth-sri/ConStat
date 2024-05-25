from .base import BaseClass
import torch
from .utils import get_max_length
    
class SingleMetric(BaseClass):
    """
    A class representing a single metric.

    This class inherits from the BaseClass and provides additional functionality for handling single metrics.

    Args:
        **kwargs: Additional keyword arguments to be passed to the BaseClass constructor.

    Attributes:
        None

    Methods:
        None
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Perplexity(SingleMetric):
    def __init__(self, model, tokenizer, **kwargs):
        """
        Initializes the Overlap class.

        Args:
            model: The model used for contamination detection.
            tokenizer: The tokenizer used for tokenizing input data.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = get_max_length(model.config)
        super().__init__(**kwargs)

    def batch_call(self, outputs, inputs=None, batch_size=1):
        """
        Calculate perplexity for a batch of outputs.

        Args:
            outputs (list): A list of output strings.
            inputs (list, optional): A list of input strings. Defaults to None.
            batch_size (int, optional): The batch size. Defaults to 1.

        Returns:
            list: A list of perplexity values for each item in the batch.
        """
        indices_with_0_length_output = []
        for i in range(len(outputs)):
            if not isinstance(outputs[i], str) or len(outputs[i]) == 0:
                indices_with_0_length_output.append(i)
        if len(indices_with_0_length_output) > 0:
            outputs_here = [outputs[i] for i in range(len(outputs)) if i not in indices_with_0_length_output]
            inputs_here = None
            if inputs is not None:
                inputs_here = [inputs[i] for i in range(len(inputs)) if i not in indices_with_0_length_output]
            perplexity = self.batch_call(outputs_here, inputs_here, batch_size)
            # arrange the topkmin list to have the same length as the outputs list
            for i in range(len(indices_with_0_length_output)):
                perplexity.insert(indices_with_0_length_output[i], 0)
            return perplexity
        # Tokenize outputs
        output_tokens = [self.tokenizer.encode(output, return_tensors='pt', add_special_tokens=False).to(self.model.device) for output in outputs]
        # Tokenize inputs if provided
        input_tokens = None
        if inputs is not None:
            input_tokens = [self.tokenizer.encode(input, return_tensors='pt').to(self.model.device) for input in inputs]

        perplexities = []
        for i in range(0, len(outputs), batch_size):
            batch_output_tokens = output_tokens[i:i+batch_size]
            # Handling input tokens for the batch
            batch_input_tokens = None
            if input_tokens is not None:
                batch_input_tokens = input_tokens[i:i+batch_size]

            # Padding tokens in the batch to have the same length
            if batch_input_tokens is not None:
                token_tensors = [torch.cat([batch_input_tokens[j], batch_output_tokens[j]], dim=-1) for j in range(len(batch_output_tokens))]
            else:
                token_tensors = batch_output_tokens
            
            # pad token tensors to get a rectangular tensor
            token_tensors_padded = torch.nn.utils.rnn.pad_sequence([token_tensor[0] for token_tensor in token_tensors], batch_first=True, 
                                                                   padding_value=self.tokenizer.pad_token_id).to(self.model.device)
            # Truncate the tokens_tensor if it is longer than the max length
            if token_tensors_padded.size(1) > self.max_length:
                token_tensors_padded = token_tensors_padded[:, :self.max_length - 1]

            # Calculate log likelihoods for the batch
            with torch.no_grad():
                outputs = self.model(input_ids=token_tensors_padded)
                logits = torch.log_softmax(outputs.logits, dim=-1)

                # Compute perplexity for each item in the batch
                for j in range(logits.shape[0]):
                    logits_index = logits[j]
                    if len(batch_output_tokens[j]) == 0:
                        perplexities.append(0)
                        continue
                    if batch_input_tokens is not None:
                        logits_index = logits_index[batch_input_tokens[j].shape[1] - 1:]
                        if logits_index.shape[0] == 0:
                            perplexities.append(10000)
                            continue
                        log_likelihood = logits_index[:-1, :].gather(1, batch_output_tokens[j][0, :logits_index.shape[0] - 1].unsqueeze(-1)).mean()
                    else:
                        log_likelihood = logits_index[:-1, :].gather(1, batch_output_tokens[j][0, 1:logits_index.shape[0]].unsqueeze(-1)).mean()
                    perplexity = torch.exp(-log_likelihood)
                    perplexities.append(perplexity.item())

        return perplexities

class Lowercase(Perplexity):
    # https://arxiv.org/pdf/2012.07805.pdf
    def __init__(self, model, tokenizer, **kwargs):
        """
        Initializes the Overlap class.

        Args:
            model: The model object used for contamination detection.
            tokenizer: The tokenizer object used for tokenizing input data.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(model, tokenizer, **kwargs)

    def batch_call(self, outputs, inputs=None, batch_size=1):
        """
        Perform a batch call to the superclass's `batch_call` method on lowercase input.

        Args:
            outputs (list): A list of outputs to be processed. Each output can be a string or an integer.
            inputs (optional): The inputs to be passed to the superclass's `batch_call` method. Defaults to None.
            batch_size (int): The batch size for processing the outputs. Defaults to 1.

        Returns:
            list: A list of perplexities calculated from the lowercased outputs.

        """
        perplexities_lower = super().batch_call([output.lower() if isinstance(output, str) else 0 for output in outputs], inputs, batch_size)
        return perplexities_lower

class TopKMin(SingleMetric):
    # https://arxiv.org/pdf/2310.16789.pdf
    def __init__(self, model, tokenizer, k=0.2, **kwargs):
        """
        Initialize the Overlap class.

        Args:
            model: The model used for contamination detection.
            tokenizer: The tokenizer used for tokenizing input data.
            k (float): The number of most unlikely tokens to consider. Defaults to 0.2.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.model = model
        self.tokenizer = tokenizer
        self.k = k
        self.max_length = get_max_length(model.config)
        super().__init__(**kwargs)

    def batch_call(self, outputs, inputs=None, batch_size=1):
        """
        Perform batch processing on a list of outputs and inputs (optional) using a specified batch size.

        Args:
            outputs (list): A list of output strings.
            inputs (list, optional): A list of input strings. Defaults to None.
            batch_size (int, optional): The batch size for processing. Defaults to 1.

        Returns:
            list: A list of topkmin values calculated for each output.

        """
        # Tokenize outputs
        indices_with_0_length_output = []
        for i in range(len(outputs)):
            if not isinstance(outputs[i], str) or len(outputs[i]) == 0:
                indices_with_0_length_output.append(i)
        if len(indices_with_0_length_output) > 0:
            outputs_here = [outputs[i] for i in range(len(outputs)) if i not in indices_with_0_length_output]
            inputs_here = None
            if inputs is not None:
                inputs_here = [inputs[i] for i in range(len(inputs)) if i not in indices_with_0_length_output]
            topkmin = self.batch_call(outputs_here, inputs_here, batch_size)
            # arrange the topkmin list to have the same length as the outputs list
            for i in range(len(indices_with_0_length_output)):
                topkmin.insert(indices_with_0_length_output[i], 0)
            return topkmin

        output_tokens = [self.tokenizer.encode(output, return_tensors='pt', add_special_tokens=False).to(self.model.device) for output in outputs]
        # Tokenize inputs if provided
        input_tokens = None
        if inputs is not None:
            input_tokens = [self.tokenizer.encode(input, return_tensors='pt').to(self.model.device) for input in inputs]

        topkmin = []
        for i in range(0, len(outputs), batch_size):
            batch_output_tokens = output_tokens[i:i+batch_size]
            # Handling input tokens for the batch
            batch_input_tokens = None
            if input_tokens is not None:
                batch_input_tokens = input_tokens[i:i+batch_size]

            # Padding tokens in the batch to have the same length
            if batch_input_tokens is not None:
                token_tensors = [torch.cat([batch_input_tokens[j], batch_output_tokens[j]], dim=-1) for j in range(len(batch_output_tokens))]
            else:
                token_tensors = batch_output_tokens
            # pad token tensors to get a rectangular tensor
            token_tensors_padded = torch.nn.utils.rnn.pad_sequence([token_tensor[0] for token_tensor in token_tensors], batch_first=True, 
                                                                   padding_value=self.tokenizer.pad_token_id).to(self.model.device)
            # Truncate the tokens_tensor if it is longer than the max length
            if token_tensors_padded.size(1) > self.max_length:
                token_tensors_padded = token_tensors_padded[:, :self.max_length - 1]

            # Calculate log likelihoods for the batch
            with torch.no_grad():
                outputs = self.model(token_tensors_padded)
                logits = torch.log_softmax(outputs.logits, dim=-1)

                # Compute perplexity for each item in the batch
                for j in range(logits.shape[0]):
                    logits_index = logits[j]
                    if len(batch_output_tokens[j]) == 0:
                        topkmin.append(0)
                        continue
                    if batch_input_tokens is not None:
                        logits_index = logits_index[batch_input_tokens[j].shape[1] - 1:]
                        if logits_index.shape[0] == 0:
                            topkmin.append(10000)
                            continue
                        log_likelihood = logits_index[:-1, :].gather(1, batch_output_tokens[j][0, :logits_index.shape[0] - 1].unsqueeze(-1))
                    else:
                        log_likelihood = logits_index[:-1, :].gather(1, batch_output_tokens[j][0, 1:logits_index.shape[0]].unsqueeze(-1))
                    # get the least likely tokens, top-k
                    top_k = int(self.k * log_likelihood.size(0))
                    if top_k == 0:
                        top_k = 1
                    least_likely_tokens = torch.topk(log_likelihood, top_k, dim=0, largest=False)[0]
                    # get the mean of the least likely tokens
                    mean = least_likely_tokens.mean(dim=0)
                    topkmin.append(mean.item())
        return topkmin