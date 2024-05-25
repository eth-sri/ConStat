import os
from loguru import logger
import aiohttp
import asyncio
import numpy as np
import json
import time

class OpenAIQuery:
    def __init__(self, model="gpt-3.5-turbo", tpm=30000, 
                 timeout=100, temperature=0, max_tokens=256, 
                 error_stop=10 ** 8, url='https://api.openai.com/v1/chat/completions',
                 is_azure=False,
                 read_cost=0.03, write_cost=0.06,
                 **kwargs) -> None:
        """
        Initialize the OpenAIQuery object.

        Args:
            model (str): The name of the model to use. Defaults to "gpt-3.5-turbo".
            tpm (int): The tokens per minute rate limit for the API. Defaults to 30000.
            timeout (int): The maximum time in seconds to wait for a response from the API. Defaults to 100.
            temperature (float): The temperature parameter for generating text. Defaults to 0.
            max_tokens (int): The maximum number of tokens to generate in the response. Defaults to 256.
            error_stop (int): The maximum number of errors to tolerate before stopping the API calls. Defaults to 10 ** 8.
            **kwargs: Additional keyword arguments to pass to the OpenAI API.

        Returns:
            None
        """
        self.model = model
        self.tpm = tpm
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.error_stop = error_stop
        self.kwargs = kwargs
        self.url = url
        self.is_azure = is_azure
        self.read_cost = read_cost
        self.write_cost = write_cost
        if is_azure:
            self.url = f'{self.url}openai/deployments/{self.model}/chat/completions?api-version=2023-07-01-preview'
        
    async def run_string_prompts(self, string_prompts):
            """
            Runs string prompts through the OpenAI model and returns the completions.

            Args:
                string_prompts (list): A list of string prompts to be processed.

            Returns:
                list: A list of completions generated by the OpenAI model.
            """
            kwarg = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "model": self.model,
            }
            openai_queries = []
            for prompt in string_prompts:
                if isinstance(prompt, str):
                    openai_queries.append({"prompt": prompt, **kwarg})
                else:
                    openai_queries.append({"messages": prompt, **kwarg})

            return await self.get_completions(openai_queries)

    async def get_completion_async(self, arguments, session):
            """
            Sends a request to the OpenAI API to get completions based on the provided arguments.

            Args:
                arguments (dict): The arguments to be sent in the request.
                session (aiohttp.ClientSession): The aiohttp client session.

            Returns:
                bytes: The response content as bytes, or None if an error occurred.
            """
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in environment variables")

            try:
                async with session.post(
                    self.url, 
                    headers={
                        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                        "Content-Type": "application/json",
                        "api-key": f"{os.environ['OPENAI_API_KEY']}"
                    },
                    json=arguments
                ) as response:
                    resp = await response.read()
                    return resp
            except Exception as e:
                logger.warning(f"Error occurred while posting to openai API: {e}. Posted: {arguments}")
                return None
        
    async def get_completions_async(self, list_arguments):
            """
            Retrieves completions asynchronously for a list of arguments.

            Args:
                list_arguments (list): A list of arguments for which completions need to be retrieved.

            Returns:
                list: A list of completions for each argument.
            """
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                ret = await asyncio.gather(*[self.get_completion_async(argument, session) for argument in list_arguments])
                
            return ret

    async def get_completions(self, list_arguments):
            """
            Retrieves completions from the OpenAI API for a list of arguments.

            Args:
                list_arguments (list): A list of arguments for which completions are requested.

            Returns:
                list: A list of completion outputs for each argument.
            """
            total_tokens = {'read': 0, 'write': 0}
            succeeded_requests = [False for _ in range(len(list_arguments))]
            outputs = [None for _ in range(len(list_arguments))]
            generated_tokens = []
            n_errors = 0
            n_parse_errors = 0
            n_new_errors = 0
            while not all(succeeded_requests) and n_errors < self.error_stop and n_parse_errors < self.error_stop:
                start_time = time.time()
                generated_tokens_last_min = sum([usage[1] for usage in generated_tokens if start_time - usage[0] < 60])
                async_requests = (self.tpm - min(generated_tokens_last_min, self.tpm)) // self.max_tokens
                if async_requests == 0:
                    time.sleep(0.2)
                    continue

                indices = np.where(np.logical_not(succeeded_requests))[0][:async_requests]
                arguments_async = [list_arguments[index] for index in indices]
                logger.debug(f"Running {len(arguments_async)} requests to openai API. tokens last minute: {generated_tokens_last_min}. percentage done: {np.count_nonzero(succeeded_requests) / len(succeeded_requests) * 100:.2f}%")
                if asyncio.get_event_loop().is_running():
                    ret = await self.get_completions_async(arguments_async)
                else:
                    ret = await asyncio.run(self.get_completions_async(arguments_async))

                for results, index in zip(ret, indices):
                    if results is not None:
                        try:
                            outputs[index] = json.loads(results)
                            if "error" not in outputs[index]:
                                succeeded_requests[index] = True
                                generated_tokens.append((start_time, outputs[index]["usage"]["total_tokens"]))
                                total_tokens['read'] += outputs[index]['usage']['prompt_tokens']
                                total_tokens['write'] += outputs[index]['usage']['completion_tokens']
                                outputs[index] = outputs[index]["choices"][0]
                            else: 
                                logger.warning(f"OpenAI API returned an error: {outputs[index]} \n On parameters {list_arguments[index]}")
                                n_errors += 1
                                n_new_errors += 1
                        except Exception:
                            logger.warning(f"OpenAI API returned invalid json: {results} \n On parameters {list_arguments[index]}")
                            n_parse_errors += 1
                    else:
                        n_errors += 1
                        n_new_errors += 1

                if n_new_errors >= 20:
                    time.sleep(10)
                    n_new_errors = 0
                        
            if n_errors >= self.error_stop or n_parse_errors >= self.error_stop:
                raise ValueError("OpenAI API returned too many errors. Stopping requests.")

            cost = total_tokens['read'] * self.read_cost / 1000
            cost += total_tokens['write'] * self.write_cost / 1000
            return outputs, cost