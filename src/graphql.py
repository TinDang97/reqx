"""
GraphQL client extension for enhanced-httpx.

This module provides specialized functionality for working with GraphQL APIs,
including support for queries, mutations, subscriptions, and handling of variables.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from .client import EnhancedClient
from .exceptions import GraphQLError

logger = logging.getLogger("enhanced_httpx.graphql")

T = TypeVar("T")


class GraphQLRequest(BaseModel):
    """A GraphQL request object."""

    query: str
    variables: Optional[Dict[str, Any]] = None
    operation_name: Optional[str] = None


class GraphQLResponse(BaseModel):
    """A GraphQL response object."""

    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[Dict[str, Any]]] = None

    @property
    def has_errors(self) -> bool:
        """Check if the response contains errors."""
        return self.errors is not None and len(self.errors) > 0

    def raise_for_errors(self):
        """Raise an exception if the response contains errors."""
        if self.has_errors:
            error_message = "; ".join(
                error.get("message", "Unknown error") for error in self.errors
            )
            raise GraphQLError(error_message, errors=self.errors, data=self.data)


class GraphQLClient:
    """
    A client for interacting with GraphQL APIs.

    This class is built on top of EnhancedClient and provides specialized
    methods for working with GraphQL APIs.
    """

    def __init__(
        self,
        endpoint: str,
        client: Optional[EnhancedClient] = None,
        headers: Dict[str, str] = None,
        **kwargs,
    ):
        """
        Initialize a GraphQL client.

        Args:
            endpoint: The GraphQL endpoint URL
            client: An optional EnhancedClient instance to use
            headers: Default headers to include with all GraphQL requests
            **kwargs: Additional arguments to pass to EnhancedClient constructor
        """
        self.endpoint = endpoint

        # Use provided client or create a new one
        if client is not None:
            self.client = client
        else:
            self.client = EnhancedClient(**kwargs)

        # Set default headers for GraphQL
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Add custom headers if provided
        if headers:
            self.headers.update(headers)

        # Keep track of document strings for query optimization
        self._document_cache = {}

    async def __aenter__(self):
        """Support for async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        await self.close()

    async def close(self):
        """Close the underlying client."""
        await self.client.close()

    async def query(
        self,
        query: str,
        variables: Dict[str, Any] = None,
        operation_name: str = None,
        parse_response: bool = True,
        response_model: Type[T] = None,
        headers: Dict[str, str] = None,
        **kwargs,
    ) -> Union[GraphQLResponse, T, Dict[str, Any]]:
        """
        Execute a GraphQL query.

        Args:
            query: The GraphQL query string
            variables: Optional variables to include with the query
            operation_name: Optional operation name for multi-operation documents
            parse_response: Whether to parse the response into a GraphQLResponse object
            response_model: Optional Pydantic model to parse data into
            headers: Optional headers to include with this request
            **kwargs: Additional arguments to pass to the request method

        Returns:
            The GraphQL response, either as a GraphQLResponse object, a Pydantic model,
            or the raw JSON data depending on the parameters
        """
        # Optimizize query if it's a string literal
        query = self._optimize_query(query)

        # Prepare request payload
        payload = {
            "query": query,
        }

        if variables:
            payload["variables"] = variables

        if operation_name:
            payload["operationName"] = operation_name

        # Merge headers
        merged_headers = {**self.headers}
        if headers:
            merged_headers.update(headers)

        # Execute request
        response = await self.client.post(
            self.endpoint, json=payload, headers=merged_headers, **kwargs
        )

        # Parse response
        json_data = response.json()

        # Handle errors in the HTTP response
        if response.status_code != 200:
            message = json_data.get(
                "message", f"GraphQL request failed with status {response.status_code}"
            )
            raise GraphQLError(message, http_status=response.status_code, response_data=json_data)

        # Return raw data if requested
        if not parse_response:
            return json_data

        # Parse into GraphQLResponse
        graphql_response = GraphQLResponse.model_validate(json_data)

        # If errors are present in GraphQL response, raise exception if strict mode
        if kwargs.get("strict", False) and graphql_response.has_errors:
            graphql_response.raise_for_errors()

        # Parse into response model if provided
        if response_model is not None:
            if graphql_response.data is None:
                raise GraphQLError("No data in GraphQL response", errors=graphql_response.errors)
            return response_model.model_validate(graphql_response.data)

        return graphql_response

    async def mutation(
        self,
        mutation: str,
        variables: Dict[str, Any] = None,
        operation_name: str = None,
        parse_response: bool = True,
        response_model: Type[T] = None,
        headers: Dict[str, str] = None,
        **kwargs,
    ) -> Union[GraphQLResponse, T, Dict[str, Any]]:
        """
        Execute a GraphQL mutation.

        This is essentially the same as a query, but is provided as a
        separate method for clarity in code and follows GraphQL conventions.

        Args:
            mutation: The GraphQL mutation string
            variables: Optional variables to include with the mutation
            operation_name: Optional operation name for multi-operation documents
            parse_response: Whether to parse the response into a GraphQLResponse object
            response_model: Optional Pydantic model to parse data into
            headers: Optional headers to include with this request
            **kwargs: Additional arguments to pass to the request method

        Returns:
            The GraphQL response, either as a GraphQLResponse object, a Pydantic model,
            or the raw JSON data depending on the parameters
        """
        # Delegate to query method since the implementation is the same
        return await self.query(
            mutation,
            variables=variables,
            operation_name=operation_name,
            parse_response=parse_response,
            response_model=response_model,
            headers=headers,
            **kwargs,
        )

    async def execute_multiple(
        self,
        operations: List[Dict[str, Any]],
        concurrency_limit: int = 5,
        parse_response: bool = True,
        **kwargs,
    ) -> List[GraphQLResponse]:
        """
        Execute multiple GraphQL operations in parallel.

        Args:
            operations: List of operation specs, each containing 'query' and optional
                      'variables' and 'operation_name'
            concurrency_limit: Maximum number of concurrent operations
            parse_response: Whether to parse responses into GraphQLResponse objects
            **kwargs: Additional arguments to pass to all queries

        Returns:
            List of GraphQL responses in the same order as the operations
        """
        tasks = []
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def _execute_operation(op):
            async with semaphore:
                query = op.get("query")
                variables = op.get("variables")
                operation_name = op.get("operation_name")

                # Optimize query
                query = self._optimize_query(query)

                return await self.query(
                    query=query,
                    variables=variables,
                    operation_name=operation_name,
                    parse_response=parse_response,
                    **kwargs,
                )

        for op in operations:
            tasks.append(_execute_operation(op))

        return await asyncio.gather(*tasks)

    def _optimize_query(self, query: str) -> str:
        """
        Optimize a GraphQL query by removing whitespace and comments.

        This is useful for reducing the size of queries sent over the wire.
        The optimization is cached to avoid redundant processing.

        Args:
            query: The GraphQL query string

        Returns:
            The optimized query string
        """
        # Check if this query is already optimized and cached
        if query in self._document_cache:
            return self._document_cache[query]

        # Don't optimize if the query is already compact
        if len(query) < 100 and "\n" not in query:
            return query

        # Remove comments (both # and /** */ style)
        query = re.sub(r"#.*$", "", query, flags=re.MULTILINE)
        query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)

        # Remove extra whitespace
        query = re.sub(r"\s+", " ", query)

        # Trim leading/trailing whitespace
        query = query.strip()

        # Cache the result
        self._document_cache[query] = query

        return query
