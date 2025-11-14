"""
SGLang client for async batch generation.

Provides efficient async batch generation for Mtq (Table→Query)
and Mqt (Query→Table) models via SGLang server.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class GenerationResult:
    """Result from a single generation request."""

    text: str
    success: bool
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class SGLangClient:
    """
    Async client for SGLang server.

    Supports batch generation with concurrent requests for efficient inference.
    """

    def __init__(
        self,
        base_url: str,
        max_concurrent: int = 32,
        timeout: float = 60.0,
    ):
        """
        Initialize SGLang client.

        Args:
            base_url: Base URL of SGLang server (e.g., http://localhost:8000)
            max_concurrent: Maximum concurrent requests
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> List[GenerationResult]:
        """
        Generate completions for a batch of prompts asynchronously.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Returns:
            List of GenerationResult objects
        """
        if not prompts:
            return []

        # Create tasks for all prompts
        tasks = [
            self._generate_single(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
            for prompt in prompts
        ]

        # Execute concurrently with semaphore limiting
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Generation failed for prompt {i}: {result}")
                final_results.append(
                    GenerationResult(
                        text="",
                        success=False,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def _generate_single(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> GenerationResult:
        """Generate completion for a single prompt."""
        async with self._semaphore:
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "text": prompt,
                        "sampling_params": {
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                            "top_p": top_p,
                        },
                    }

                    if stop:
                        payload["sampling_params"]["stop"] = stop

                    async with session.post(
                        f"{self.base_url}/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            return GenerationResult(
                                text="",
                                success=False,
                                error=f"HTTP {response.status}: {error_text}",
                            )

                        data = await response.json()
                        generated_text = data.get("text", "")

                        # Debug: Log if empty generation
                        if not generated_text:
                            logger.warning(f"Empty generation from SGLang. Response keys: {list(data.keys())}")
                            logger.warning(f"Full response: {data}")

                        return GenerationResult(
                            text=generated_text,
                            success=True,
                            raw_response=data,
                        )

            except asyncio.TimeoutError:
                return GenerationResult(
                    text="",
                    success=False,
                    error="Request timeout",
                )
            except Exception as e:
                logger.exception(f"Unexpected error during generation: {e}")
                return GenerationResult(
                    text="",
                    success=False,
                    error=str(e),
                )

    def generate_batch_sync(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> List[GenerationResult]:
        """
        Synchronous wrapper for generate_batch.

        Useful for non-async contexts.
        """
        return asyncio.run(
            self.generate_batch(
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
        )

    async def health_check(self) -> bool:
        """Check if the SGLang server is healthy."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


class DualSGLangClient:
    """
    Manages two SGLang clients for Mtq and Mqt models.

    Provides convenient interface for cycle training where we need
    both Table→Query and Query→Table generation.
    """

    def __init__(
        self,
        mtq_url: str,
        mqt_url: str,
        max_concurrent: int = 32,
        timeout: float = 60.0,
    ):
        """
        Initialize dual clients.

        Args:
            mtq_url: URL for Mtq model (Table→Query)
            mqt_url: URL for Mqt model (Query→Table)
            max_concurrent: Maximum concurrent requests per client
            timeout: Request timeout in seconds
        """
        self.mtq_client = SGLangClient(mtq_url, max_concurrent, timeout)
        self.mqt_client = SGLangClient(mqt_url, max_concurrent, timeout)

        logger.info(f"Initialized dual SGLang clients: Mtq={mtq_url}, Mqt={mqt_url}")

    async def generate_queries(
        self,
        table_prompts: List[str],
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> List[GenerationResult]:
        """Generate pseudo-queries from tables using Mtq."""
        return await self.mtq_client.generate_batch(
            prompts=table_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop=None,  # Don't use stop tokens - let model generate complete output
        )

    async def generate_tables(
        self,
        query_prompts: List[str],
        max_new_tokens: int = 196,
        temperature: float = 0.0,
    ) -> List[GenerationResult]:
        """Generate pseudo-tables from queries using Mqt."""
        return await self.mqt_client.generate_batch(
            prompts=query_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop=None,  # Don't use stop tokens - let model generate complete JSON
        )

    async def health_check_both(self) -> Dict[str, bool]:
        """Check health of both servers."""
        mtq_health, mqt_health = await asyncio.gather(
            self.mtq_client.health_check(),
            self.mqt_client.health_check(),
        )
        return {
            "mtq": mtq_health,
            "mqt": mqt_health,
            "all_healthy": mtq_health and mqt_health,
        }
